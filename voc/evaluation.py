"""Evaluation harness. Three questions, answered with numbers:

1. Extraction: does it find the right aspects and sentiments?  (vs. hand-labelled gold set)
2. Themes:     are theme labels/summaries faithful to the quotes behind them?  (LLM-as-judge + cohesion)
3. SQL:        does Claude write correct SQL against our schema, and does the guard block bad queries?
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import yaml

from voc.config import ROOT, settings
from voc.db import connect, rows
from voc.llm import get_llm, usage, use_llm
from voc.pipeline.extract import extract_reviews
from voc.schemas import ASPECTS, JudgeVerdict, SQLAnswer
from voc.serve import service
from voc.serve.sql_guard import UnsafeSQL, validate


def _write(name: str, md: str, data: dict) -> Path:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    p = settings.reports_dir / f"{name}.md"
    p.write_text(md, encoding="utf-8")
    (settings.reports_dir / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return p


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


# ------------------------------------------------------------------ 1. extraction

def eval_extraction(gold_path: Path, backend: str) -> dict:
    gold = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    t0 = time.time()
    before = usage.as_dict()["est_cost_usd"]
    preds, xstats = extract_reviews(
        [{"review_id": g["review_id"], "product_id": g["product_id"], "text": g["text"]} for g in gold], backend)
    elapsed = time.time() - t0
    pmap = {p.review_id: p for p in preds}

    rel = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    joint = {"tp": 0, "fp": 0, "fn": 0}
    sent_ok = sent_n = 0
    errors = []
    for g in gold:
        p = pmap.get(g["review_id"])
        p_rel = bool(p and p.is_relevant)
        if g["is_relevant"]:
            rel["tp" if p_rel else "fn"] += 1
        else:
            rel["fp" if p_rel else "tn"] += 1
        g_asp = {a["aspect"]: a["sentiment"] for a in g["aspects"]}
        p_asp = {m.aspect: m.sentiment for m in (p.aspects if p else [])}
        for a in set(g_asp) | set(p_asp):
            if a in g_asp and a in p_asp:
                per[a]["tp"] += 1
                sent_n += 1
                sent_ok += g_asp[a] == p_asp[a]
            elif a in p_asp:
                per[a]["fp"] += 1
            else:
                per[a]["fn"] += 1
        gj, pj = set(g_asp.items()), set(p_asp.items())
        joint["tp"] += len(gj & pj)
        joint["fp"] += len(pj - gj)
        joint["fn"] += len(gj - pj)
        if gj != pj and len(errors) < 25:
            errors.append({"review_id": g["review_id"], "text": g["text"], "gold": sorted(gj), "pred": sorted(pj)})

    per_aspect = {a: dict(zip(("precision", "recall", "f1"), _prf(**per[a])), support=per[a]["tp"] + per[a]["fn"])
                  for a in ASPECTS if a in per}
    micro = _prf(sum(v["tp"] for v in per.values()), sum(v["fp"] for v in per.values()),
                 sum(v["fn"] for v in per.values()))
    macro_f1 = round(sum(v["f1"] for v in per_aspect.values()) / len(per_aspect), 3) if per_aspect else 0.0
    rel_acc = round((rel["tp"] + rel["tn"]) / len(gold), 3)
    cost = usage.as_dict()["est_cost_usd"] - before
    result = {
        "backend": backend, "gold_size": len(gold), "relevance_accuracy": rel_acc, "relevance_confusion": rel,
        "aspect_detection_micro": dict(zip(("precision", "recall", "f1"), micro)), "aspect_detection_macro_f1": macro_f1,
        "aspect_plus_sentiment_micro": dict(zip(("precision", "recall", "f1"), _prf(**joint))),
        "sentiment_accuracy_on_matched": round(sent_ok / sent_n, 3) if sent_n else None,
        "span_verification_rate": xstats.get("span_verification_rate"),
        "seconds": round(elapsed, 1),
        "est_cost_usd": round(cost, 4) if backend == "anthropic" else 0.0,
        "per_aspect": per_aspect, "sample_errors": errors,
    }
    md = [f"# Extraction eval · backend `{backend}` · {len(gold)} gold reviews", "",
          f"Gold file: `{gold_path.name}`", "",
          "| Metric | Value |", "|---|---|",
          f"| Relevance accuracy | {rel_acc} |",
          f"| Aspect detection P / R / F1 (micro) | {' / '.join(map(str, micro))} |",
          f"| Aspect detection F1 (macro) | {macro_f1} |",
          f"| Aspect + sentiment F1 (micro) | {result['aspect_plus_sentiment_micro']['f1']} |",
          f"| Sentiment accuracy (matched aspects) | {result['sentiment_accuracy_on_matched']} |",
          f"| Evidence spans verified verbatim | {result['span_verification_rate']} |",
          f"| Wall time | {result['seconds']} s |",
          f"| Est. cost | ${result['est_cost_usd']} |", "",
          "## Per aspect", "", "| Aspect | P | R | F1 | Support |", "|---|---|---|---|---|"]
    md += [f"| {a} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |" for a, v in per_aspect.items()]
    md += ["", "## Sample disagreements", ""]
    md += [f"- `{e['review_id']}`: {e['text'][:140]}  \n  gold {e['gold']} · pred {e['pred']}" for e in errors[:10]]
    if "synthetic" in gold_path.name:
        md.insert(1, "\n> Synthetic gold set: these numbers only prove the harness runs. Use your hand-labelled set for real results.\n")
    result["report"] = str(_write(f"eval_extraction_{backend}", "\n".join(md), result))
    return result


# ------------------------------------------------------------------ 2. themes

JUDGE_SYSTEM = """You audit consumer-insight summaries for faithfulness. Given a theme label, a summary, and
the verbatim consumer quotes it was built from, check every factual claim in the label and summary.
A claim is supported only if the quotes directly say it. Invented causes, numbers, ingredients,
demographics or time frames are unsupported."""


def eval_themes(limit: int = 40) -> dict:
    con = connect(settings.db_path, read_only=True)
    themes = rows(con, "SELECT * FROM themes ORDER BY n_reviews DESC LIMIT ?", [limit])
    cohesion = round(sum(t["cohesion"] for t in themes) / len(themes), 3) if themes else None
    result = {"themes_evaluated": len(themes), "mean_cohesion": cohesion, "judge": None}
    if themes and use_llm():
        verdicts = []
        for t in themes:
            quotes = [r[0] for r in con.execute(
                "SELECT evidence_span FROM theme_evidence WHERE theme_id = ? ORDER BY rank LIMIT 12",
                [t["theme_id"]]).fetchall()]
            v = get_llm().structured(JudgeVerdict, (
                f"Label: {t['label']}\nSummary: {t['summary']}\nQuotes:\n" + "\n".join(f"- {q}" for q in quotes)
            ), system=JUDGE_SYSTEM, tier="smart", max_tokens=600)
            verdicts.append({"theme_id": t["theme_id"], "label": t["label"], **v.model_dump()})
        result["judge"] = {
            "mean_faithfulness": round(sum(v["score"] for v in verdicts) / len(verdicts), 3),
            "fully_supported_rate": round(sum(v["supported"] for v in verdicts) / len(verdicts), 3),
            "failures": [v for v in verdicts if not v["supported"]],
        }
    con.close()
    md = ["# Theme eval", "", f"Themes evaluated: {len(themes)}",
          f"Mean cohesion (avg cosine of quotes to theme centroid): {cohesion}", ""]
    if result["judge"]:
        j = result["judge"]
        md += [f"LLM-judge mean faithfulness: **{j['mean_faithfulness']}**",
               f"Themes with every claim supported: **{j['fully_supported_rate']:.0%}**", "", "## Failures", ""]
        md += [f"- **{f['label']}**: {', '.join(f['unsupported_claims'])}" for f in j["failures"]] or ["None."]
    else:
        md.append("LLM-judge skipped (needs VOC_LLM_BACKEND=anthropic). Spot-check 10 themes by hand too.")
    result["report"] = str(_write("eval_themes", "\n".join(md), result))
    return result


# ------------------------------------------------------------------ 3. SQL

ATTACKS = [
    "DROP TABLE reviews",
    "DELETE FROM tickets",
    "SELECT * FROM raw_reviews",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT getenv('ANTHROPIC_API_KEY')",
    "SELECT 1; DROP TABLE themes",
    "COPY reviews TO 'out.csv'",
    "ATTACH 'other.db' AS o",
    "INSERT INTO tickets SELECT * FROM tickets",
    "SELECT * FROM information_schema.tables",
]

SQL_SYSTEM = """You write a single DuckDB SELECT query that answers the question using only the given schema.
Return only the SQL in the `sql` field. Follow the notes in the schema exactly."""


def _canon(result: dict) -> list[tuple]:
    out = []
    for r in result.get("rows", []):
        vals = []
        for v in r.values():
            vals.append(round(v, 2) if isinstance(v, float) else v)
        out.append(tuple(str(v) for v in vals))
    return sorted(out)


def eval_sql(questions_path: Path) -> dict:
    blocked = []
    for q in ATTACKS:
        try:
            validate(q)
            blocked.append({"sql": q, "blocked": False})
        except UnsafeSQL as e:
            blocked.append({"sql": q, "blocked": True, "reason": str(e)})
    result = {"guard_blocked": sum(b["blocked"] for b in blocked), "guard_total": len(ATTACKS),
              "guard_details": blocked, "nl2sql": None}

    questions = yaml.safe_load(questions_path.read_text())["questions"]
    gold_ok = [q for q in questions if "error" not in service.run_metrics_sql(q["gold_sql"])]
    result["gold_queries_valid"] = f"{len(gold_ok)}/{len(questions)}"
    if use_llm():
        schema = json.dumps(service.get_schema(), indent=1)
        details, correct = [], 0
        for q in gold_ok:
            ans = get_llm().structured(SQLAnswer, f"Schema:\n{schema}\n\nQuestion: {q['question']}",
                                       system=SQL_SYSTEM, tier="smart", max_tokens=800)
            got = service.run_metrics_sql(ans.sql)
            ok = "error" not in got and _canon(got) == _canon(service.run_metrics_sql(q["gold_sql"]))
            correct += ok
            details.append({"question": q["question"], "generated_sql": ans.sql, "correct": ok,
                            "error": got.get("error")})
        result["nl2sql"] = {"execution_accuracy": round(correct / len(gold_ok), 3) if gold_ok else None,
                            "n": len(gold_ok), "details": details}

    md = ["# SQL eval", "", f"Guard blocked **{result['guard_blocked']}/{result['guard_total']}** attack queries.",
          f"Gold queries that run: {result['gold_queries_valid']}", ""]
    md += [f"- {'✅' if b['blocked'] else '❌'} `{b['sql']}`" for b in blocked]
    if result["nl2sql"]:
        n = result["nl2sql"]
        md += ["", f"## NL→SQL execution accuracy: **{n['execution_accuracy']}** on {n['n']} questions", ""]
        md += [f"- {'✅' if d['correct'] else '❌'} {d['question']}" for d in n["details"]]
    else:
        md += ["", "NL→SQL accuracy skipped (needs VOC_LLM_BACKEND=anthropic)."]
    result["report"] = str(_write("eval_sql", "\n".join(md), result))
    return result


DEFAULT_GOLD = ROOT / "evals/gold_labels.jsonl"
DEMO_GOLD = ROOT / "evals/demo_gold_synthetic.jsonl"
SQL_QUESTIONS = ROOT / "evals/sql_questions.yaml"
