"""Stage 2: aspect-level sentiment extraction with verbatim-evidence verification.

Two interchangeable backends:
  - anthropic: Claude with a strict Pydantic schema (forced tool use)
  - baseline:  a transparent keyword/lexicon extractor. Needs no API key, and gives the eval a floor
               to beat, so you can show how much the LLM actually adds.

Every evidence_span is checked in code to be a verbatim substring of the review. Spans that fail
are dropped and counted. This is the main hallucination guardrail: no theme can cite text that
a consumer didn't write.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from voc.config import product_map, settings
from voc.db import now, run_step
from voc.llm import get_llm, usage, use_llm
from voc.schemas import ASPECTS, AspectMention, BatchExtraction, ReviewExtraction

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- span verification


def verify_span(text: str, span: str) -> str | None:
    """Return the span exactly as it appears in `text`, or None if it isn't really there.

    Tolerates differences in case and whitespace only. Paraphrases, translations and
    "fixed" typos all fail, which is the point.
    """
    span = (span or "").strip().strip('"“”\'').strip()
    span = re.sub(r"^(\.\.\.|…)|(\.\.\.|…)$", "", span).strip()
    if len(span) < 3:
        return None
    if span in text:
        return span
    tokens = span.split()
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return text[m.start():m.end()] if m else None


# ---------------------------------------------------------------- baseline extractor

LEXICON = {
    "efficacy": ["hair fall", "hairfall", "hair growth", "baal", "oil control", "oily", "acne", "pimple",
                 "tan", "glow", "brighten", "dandruff", "result", "effective", "difference", "fark",
                 "protection", "tanning", "clear skin", "works", "worked", "no change"],
    "skin_reaction": ["breakout", "broke me out", "broke out", "rash", "itch", "burn", "irritat", "allerg",
                      "redness", "dryness", "stinging", "dane", "khujli", "jalan", "dry"],
    "fragrance": ["smell", "fragrance", "scent", "khushboo", "perfume"],
    "texture": ["texture", "white cast", "greasy", "sticky", "lather", "foam", "consistency",
                "lightweight", "watery", "chipchip", "heavy"],
    "packaging": ["pump", "cap ", "cap,", "leak", "tube", "packaging", "nozzle", "seal", "bottle broke",
                  "broken bottle", "bottle leak"],
    "price_value": ["price", "expensive", "cheap", "worth", "value for money", "paisa", "paise",
                    "overpriced", "costly", "affordable", "mehenga", "mehnga", "sasta"],
    "authenticity": ["fake", "duplicate", "expired", "expiry", "counterfeit", "tampered", "nakli"],
    "delivery": ["delivery", "delivered", "shipping", "courier", "blinkit", "zepto", "instamart",
                 "box was damaged", "damaged in transit"],
}
POS_PHRASES = ["no white cast", "no breakout", "reduced", "worth it", "value for money", "no irritation",
               "controlled", "less hair fall", "kam ho gaya", "kam hua"]
NEG_PHRASES = ["no difference", "no change", "koi fark nahi", "increased", "stopped working", "broke me out",
               "worse", "waste of money", "not worth", "white cast"]
POS = ["good", "great", "love", "amazing", "best", "works", "worked", "accha", "acha", "achha", "badhiya",
       "mast", "nice", "soft", "smooth", "lightweight", "affordable", "worth", "glow", "clear", "pleasant",
       "fresh", "perfect", "helped", "recommend"]
NEG = ["bad", "worst", "waste", "bekar", "bakwas", "didn't", "doesn't", "nahi", "nhi", "broke", "leak",
       "fake", "expired", "itch", "burn", "rash", "breakout", "irritat", "greasy", "sticky", "expensive",
       "overpriced", "damaged", "kharab", "allerg", "strong", "too much", "dry", "dane", "khujli", "jalan",
       "disappoint", "never again", "costly", "mehenga"]
SAFETY = ["rash", "burn", "allerg", "itch", "khujli", "jalan", "swelling", "fake", "expired", "tampered", "nakli"]
CLAUSE_RE = re.compile(r"[^.!?;\n]+")
SPLIT_RE = re.compile(r"\s+(?:but|lekin|par|however)\s+", re.IGNORECASE)


def _clauses(text: str):
    for m in CLAUSE_RE.finditer(text):
        start = m.start()
        chunk = m.group()
        pos = 0
        for sm in list(SPLIT_RE.finditer(chunk)) + [None]:
            end = sm.start() if sm else len(chunk)
            piece = chunk[pos:end]
            stripped = piece.strip()
            if stripped:
                offset = start + pos + piece.index(stripped)
                yield text[offset:offset + len(stripped)]
            if sm:
                pos = sm.end()


def _polarity(clause: str) -> str:
    c = clause.lower()
    pos = sum(2 for p in POS_PHRASES if p in c)
    neg = sum(2 for p in NEG_PHRASES if p in c and not any(pp in c for pp in POS_PHRASES if p in pp))
    pos += sum(1 for w in POS if w in c)
    neg += sum(1 for w in NEG if w in c)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def baseline_extract(review_id: str, text: str, product: dict | None) -> ReviewExtraction:
    mentions, seen = [], set()
    for clause in _clauses(text):
        c = clause.lower()
        for aspect, words in LEXICON.items():
            if aspect in seen or not any(w in c for w in words):
                continue
            sent = _polarity(clause)
            severity = 1
            if sent == "negative":
                severity = 3 if any(s in c for s in SAFETY) and aspect in ("skin_reaction", "authenticity") else 2
            mentions.append(AspectMention(aspect=aspect, sentiment=sent, severity=severity, evidence_span=clause))
            seen.add(aspect)
    on_topic = bool(product) and any(t in text.lower() for t in product.get("match_terms", []))
    relevant = bool(mentions) or on_topic
    if text.strip().endswith("?") and not mentions:
        relevant = False
    counts = {s: sum(m.sentiment == s for m in mentions) for s in ("positive", "negative")}
    overall = "neutral"
    if counts["negative"] > counts["positive"]:
        overall = "negative"
    elif counts["positive"] > counts["negative"]:
        overall = "positive"
    return ReviewExtraction(review_id=review_id, is_relevant=relevant, overall_sentiment=overall,
                            aspects=mentions if relevant else [])


# ---------------------------------------------------------------- LLM extractor

SYSTEM = f"""You analyse Indian consumer comments about beauty and personal-care products for a brand's
Voice-of-Consumer team. Comments may be English, Hindi, or Hinglish (romanised Hindi mixed with English).

For each comment decide:
1. is_relevant: true only if it gives a specific or first-hand opinion/experience about THE NAMED PRODUCT.
   Questions to a YouTuber, generic praise of the creator, spam, and talk about other products are NOT relevant.
2. overall_sentiment toward the product.
3. aspects: zero or more of {", ".join(ASPECTS)}.
   - efficacy: whether it delivers its promise (hair fall, oil control, acne, tan, sun protection...)
   - skin_reaction: breakouts, rashes, itching, burning, dryness, irritation, allergy
   - texture covers white cast, greasiness, lather, feel. packaging covers pumps, caps, leaks.
   - authenticity: suspected fake, duplicate, expired, tampered products.
   Use at most one mention per aspect per comment. If a comment has no aspect, return an empty list.
4. severity: 1 minor, 2 clear dissatisfaction, 3 safety or trust issue (allergic reaction, burning,
   suspected fake/expired). Positive mentions are severity 1.

evidence_span MUST be copied character-for-character from the comment: a contiguous substring, in the
original language, with the original spelling. Never translate, summarise or correct it. Spans are
verified automatically and any span that isn't verbatim is discarded.

Return exactly one result per input comment, with the same review_id."""


def _llm_batch(batch: list[dict], products: dict[str, dict]) -> list[ReviewExtraction]:
    items = []
    for r in batch:
        p = products.get(r["product_id"], {})
        items.append({
            "review_id": r["review_id"],
            "product": p.get("name", r["product_id"]),
            "context": (r.get("parent_context") or "")[:200],
            "comment": r["text"],
        })
    prompt = "Analyse these comments:\n\n" + json.dumps(items, ensure_ascii=False, indent=1)
    out = get_llm().structured(BatchExtraction, prompt, system=SYSTEM, tier="fast", max_tokens=8000)
    return out.results


FATAL_ERRORS = {"AuthenticationError", "PermissionDeniedError", "NotFoundError", "LLMError"}


def extract_reviews(reviews: list[dict], backend: str | None = None) -> tuple[list[ReviewExtraction], dict]:
    """Extract + verify. `reviews` need review_id, product_id, text (and optional parent_context)."""
    backend = backend or settings.llm_backend
    products = product_map()
    by_id = {r["review_id"]: r for r in reviews}
    results: dict[str, ReviewExtraction] = {}

    if backend == "baseline":
        for r in reviews:
            results[r["review_id"]] = baseline_extract(r["review_id"], r["text"], products.get(r["product_id"]))
    else:
        size = settings.extract_batch_size
        batches = [reviews[i:i + size] for i in range(0, len(reviews), size)]

        def run_batch(b):
            try:
                return _llm_batch(b, products)
            except Exception as e:
                if type(e).__name__ in FATAL_ERRORS:  # bad key / model name: stop now, don't retry 1-by-1
                    raise
                log.warning("batch failed (%s); will retry items individually", e)
                return []

        with ThreadPoolExecutor(max_workers=settings.llm_workers) as pool:
            for res in pool.map(run_batch, batches):
                for x in res:
                    if x.review_id in by_id:
                        results[x.review_id] = x
        for rid in [rid for rid in by_id if rid not in results]:  # retry anything the model skipped
            try:
                for x in _llm_batch([by_id[rid]], products):
                    if x.review_id == rid:
                        results[rid] = x
            except Exception as e:
                if type(e).__name__ in FATAL_ERRORS:
                    raise
                log.warning("giving up on %s: %s", rid, e)

    stats = {"backend": backend, "requested": len(reviews), "returned": len(results),
             "spans_total": 0, "spans_verified": 0}
    verified: list[ReviewExtraction] = []
    for rid, x in results.items():
        text = by_id[rid]["text"]
        keep = []
        for m in (x.aspects if x.is_relevant else []):
            stats["spans_total"] += 1
            span = verify_span(text, m.evidence_span)
            if span:
                stats["spans_verified"] += 1
                keep.append(m.model_copy(update={"evidence_span": span}))
        verified.append(x.model_copy(update={"aspects": keep}))
    stats["span_verification_rate"] = round(stats["spans_verified"] / stats["spans_total"], 3) if stats["spans_total"] else None
    return verified, stats


def run(con, limit: int | None = None) -> dict:
    model = "rules" if settings.llm_backend == "baseline" else settings.fast_model
    with run_step(con, "extract") as (_, stats):
        q = """
            SELECT r.review_id, r.product_id, r.text, r.created_at, raw.parent_context
            FROM reviews r
            LEFT JOIN extractions e USING (review_id)
            LEFT JOIN raw_reviews raw USING (review_id)
            WHERE e.review_id IS NULL
            ORDER BY r.review_id
        """ + (f" LIMIT {int(limit)}" if limit else "")
        cur = con.execute(q)
        cols = [d[0] for d in cur.description]
        pending = [dict(zip(cols, r)) for r in cur.fetchall()]
        if not pending:
            stats["note"] = "nothing to extract"
            return stats
        before = usage.as_dict()
        results, xstats = extract_reviews(pending, settings.llm_backend)
        created = {r["review_id"]: r["created_at"] for r in pending}
        pid = {r["review_id"]: r["product_id"] for r in pending}
        ts = now()
        con.executemany("INSERT INTO extractions VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (x.review_id, pid[x.review_id], x.is_relevant, x.overall_sentiment, settings.llm_backend, model, ts)
            for x in results
        ])
        con.executemany("INSERT INTO aspects VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (x.review_id, pid[x.review_id], m.aspect, m.sentiment, m.severity, m.evidence_span, created[x.review_id])
            for x in results for m in x.aspects
        ])
        stats.update(xstats)
        stats["relevant"] = sum(x.is_relevant for x in results)
        if use_llm():
            after = usage.as_dict()
            stats["est_cost_usd"] = round(after["est_cost_usd"] - before["est_cost_usd"], 4)
            stats["cost_per_1k_reviews_usd"] = round(stats["est_cost_usd"] / len(pending) * 1000, 3)
    return stats
