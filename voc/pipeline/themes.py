"""Stage 3: discover themes inside each (product, aspect, polarity) group.

We embed the evidence spans (not whole reviews), so a review that praises the fragrance but
complains about the pump contributes to two different themes. Clusters are found with HDBSCAN,
which doesn't need the number of themes up front and leaves out stray one-offs as noise.
Each theme keeps the full list of review IDs behind it, so every claim can be traced back.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from voc.config import product_map
from voc.db import new_id, now, run_step
from voc.embeddings import embed
from voc.llm import get_llm, use_llm
from voc.pipeline import tracking
from voc.schemas import ThemeLabel

log = logging.getLogger(__name__)

MIN_THEME_REVIEWS = 3      # below this, a "theme" is an anecdote
SINGLE_CLUSTER_BELOW = 12  # small groups are labelled as one theme rather than over-split
REPRESENTATIVE_K = 10


def cluster(vecs: np.ndarray) -> np.ndarray:
    n = len(vecs)
    if n < SINGLE_CLUSTER_BELOW:
        return np.zeros(n, dtype=int)
    from sklearn.cluster import HDBSCAN

    labels = HDBSCAN(min_cluster_size=max(MIN_THEME_REVIEWS, round(n * 0.08)), min_samples=2,
                     copy=True).fit_predict(vecs)
    if (labels >= 0).sum() == 0:
        return np.zeros(n, dtype=int)
    return labels


def _baseline_label(aspect: str, polarity: str, spans: list[str]) -> ThemeLabel:
    from sklearn.feature_extraction.text import TfidfVectorizer

    pretty = aspect.replace("_", " ")
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=500)
        m = vec.fit_transform(spans)
        scores = np.asarray(m.sum(axis=0)).ravel()
        names = vec.get_feature_names_out()
        terms, used = [], set()
        for i in scores.argsort()[::-1]:  # skip terms that just repeat words already chosen
            words = set(names[i].split())
            if words & used:
                continue
            terms.append(names[i])
            used |= words
            if len(terms) == 3:
                break
    except ValueError:
        terms = []
    label = f"{pretty.capitalize()} ({polarity}): {', '.join(terms)}" if terms else f"{pretty.capitalize()} ({polarity})"
    return ThemeLabel(label=label[:80], summary=f'Representative quote: "{spans[0][:160]}"')


LABEL_SYSTEM = """You name consumer-feedback themes for a beauty brand's product, R&D and marketing teams.
You are given verbatim consumer quotes that were clustered together. Write a specific label (max 8 words)
and a one-sentence summary. Every claim must be directly supported by the quotes. Do not invent causes,
numbers, ingredients or demographics. Prefer concrete wording ("Pump jams after a few uses") over vague
wording ("Packaging issues"). Quotes may be in Hinglish; write the label and summary in English."""


def _llm_label(product: str, aspect: str, polarity: str, spans: list[str], n_reviews: int) -> ThemeLabel:
    quotes = "\n".join(f"- {s}" for s in spans)
    prompt = (f"Product: {product}\nAspect: {aspect}\nPolarity: {polarity}\n"
              f"Reviews in this theme: {n_reviews}\nMost representative quotes:\n{quotes}")
    return get_llm().structured(ThemeLabel, prompt, system=LABEL_SYSTEM, tier="smart", max_tokens=400)


def run(con) -> dict:
    products = product_map()
    with run_step(con, "themes") as (run_id, stats):
        rows = con.execute("""
            SELECT a.product_id, a.aspect, a.sentiment, a.review_id, a.evidence_span
            FROM aspects a JOIN extractions e USING (review_id)
            WHERE e.is_relevant AND a.sentiment IN ('positive', 'negative')
        """).fetchall()
        groups: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
        for pid, aspect, sent, rid, span in rows:
            groups[(pid, aspect, sent)].append((rid, span))

        # Themes are a snapshot: rebuild them from scratch on every run.
        con.execute("DELETE FROM theme_evidence")
        con.execute("DELETE FROM theme_metrics")
        con.execute("DELETE FROM themes")

        made, noise = 0, 0
        for (pid, aspect, sent), items in sorted(groups.items()):
            if len({rid for rid, _ in items}) < MIN_THEME_REVIEWS:
                continue
            spans = [s for _, s in items]
            vecs = embed(spans)
            labels = cluster(vecs)
            noise += int((labels < 0).sum())
            for c in sorted(set(labels) - {-1}):
                idx = np.where(labels == c)[0]
                members = [items[i] for i in idx]
                if len({rid for rid, _ in members}) < MIN_THEME_REVIEWS:
                    continue
                centroid = vecs[idx].mean(axis=0)
                centroid /= (np.linalg.norm(centroid) or 1.0)
                sims = vecs[idx] @ centroid
                order = np.argsort(-sims)
                ranked = [members[i] for i in order]
                rep = []
                for _, s in ranked:  # distinct quotes only
                    if s not in rep:
                        rep.append(s)
                    if len(rep) == REPRESENTATIVE_K:
                        break
                n_reviews = len({rid for rid, _ in members})
                pname = products.get(pid, {}).get("name", pid)
                try:
                    lab = _llm_label(pname, aspect, sent, rep, n_reviews) if use_llm() else _baseline_label(aspect, sent, rep)
                except Exception as e:
                    log.warning("labelling failed (%s); using baseline label", e)
                    lab = _baseline_label(aspect, sent, rep)
                tid = new_id("th_")
                con.execute("INSERT INTO themes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                    tid, run_id, pid, aspect, sent, lab.label, lab.summary, n_reviews, len(members),
                    float(sims.mean()), now(),
                ])
                con.executemany("INSERT INTO theme_evidence VALUES (?, ?, ?, ?)",
                                [(tid, rid, s, r) for r, (rid, s) in enumerate(ranked)])
                made += 1
        stats.update({"groups": len(groups), "themes": made, "noise_mentions": noise})
        stats["tracking"] = tracking.link(con, run_id)
    return stats
