"""Core tests. Run with `pytest -q`. No API keys or network needed."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from voc.config import ROOT, settings
from voc.pipeline.clean import detect_lang, scrub
from voc.pipeline.extract import baseline_extract, verify_span
from voc.serve.sql_guard import UnsafeSQL, validate


# ---------------------------------------------------------------- span verification (hallucination guard)

def test_verify_span_exact_and_whitespace_tolerant():
    text = "Pump stopped working  halfway through the bottle."
    assert verify_span(text, "Pump stopped working halfway") == "Pump stopped working  halfway"
    assert verify_span(text, "pump STOPPED working") == "Pump stopped working"


@pytest.mark.parametrize("span", [
    "The pump broke after a few uses",      # paraphrase
    "Pump stopped working half way",        # "fixed" spelling
    "pump",                                 # fine, but too short to be useful? 4 chars is allowed
])
def test_verify_span_rejects_paraphrase(span):
    text = "Pump stopped working halfway through the bottle."
    if span == "pump":
        assert verify_span(text, span) == "Pump"
    else:
        assert verify_span(text, span) is None


def test_verify_span_hinglish_must_stay_untranslated():
    text = "Koi fark nahi pada hair fall mein, paisa waste ho gaya."
    assert verify_span(text, "koi fark nahi pada hair fall mein") is not None
    assert verify_span(text, "No difference in hair fall") is None


# ---------------------------------------------------------------- SQL guard

@pytest.mark.parametrize("sql", [
    "DROP TABLE reviews",
    "DELETE FROM tickets",
    "SELECT * FROM raw_reviews",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT getenv('HOME')",
    "SELECT 1; DROP TABLE themes",
    "ATTACH 'x.db' AS x",
    "COPY reviews TO 'out.csv'",
    "SELECT * FROM information_schema.tables",
    "UPDATE tickets SET status = 'approved'",
])
def test_sql_guard_blocks(sql):
    with pytest.raises(UnsafeSQL):
        validate(sql)


@pytest.mark.parametrize("sql", [
    "SELECT product_id, count(*) FROM reviews GROUP BY 1",
    "WITH x AS (SELECT * FROM aspects WHERE sentiment = 'negative') SELECT aspect, count(*) FROM x GROUP BY 1",
    "SELECT label FROM themes UNION ALL SELECT theme_label FROM tickets",
])
def test_sql_guard_allows_reads_and_caps_rows(sql):
    safe = validate(sql, max_rows=50)
    assert safe.endswith("LIMIT 50")


# ---------------------------------------------------------------- cleaning + baseline

def test_scrub_removes_pii():
    out = scrub("Mail me at a.b@gmail.com or call 98765 43210, see https://x.co @priya_k")
    assert "gmail" not in out and "98765" not in out and "https" not in out and "priya" not in out


def test_detect_lang():
    assert detect_lang("Bahut accha hai, hair fall kam ho gaya") == "hinglish"
    assert detect_lang("Works well for oily skin") == "en"
    assert detect_lang("यह शैम्पू बहुत अच्छा है और बाल झड़ना कम हुआ") == "hi"


def test_baseline_extract_finds_aspects_with_verbatim_spans():
    text = "Smells lovely and fresh. But the pump stopped working after a week."
    x = baseline_extract("r1", text, {"match_terms": []})
    aspects = {m.aspect: m for m in x.aspects}
    assert aspects["fragrance"].sentiment == "positive"
    assert aspects["packaging"].sentiment == "negative"
    assert all(m.evidence_span in text for m in x.aspects)


def test_baseline_marks_creator_questions_irrelevant():
    assert not baseline_extract("r2", "Which camera do you use for these videos?", {"match_terms": []}).is_relevant


# ---------------------------------------------------------------- end-to-end on synthetic data

@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    """Run the full offline pipeline once into a temp DB."""
    from voc.collectors.file_import import import_file
    from voc.db import connect, sync_products
    from voc.graph import insight_graph
    from voc.pipeline import clean, digest, extract, themes

    tmp = tmp_path_factory.mktemp("voc")
    old = (settings.db_path, settings.reports_dir, settings.llm_backend, settings.embed_backend)
    settings.db_path, settings.reports_dir = tmp / "t.duckdb", tmp / "reports"
    settings.llm_backend, settings.embed_backend = "baseline", "tfidf"
    sample = ROOT / "data/sample/synthetic_reviews.jsonl"
    if not sample.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic.py")], check=True)
    con = connect()
    sync_products(con)
    import_file(con, sample, "synthetic", url_col=None, id_col="review_id")
    clean.run(con)
    extract.run(con)
    themes.run(con)
    decisions = []

    def reviewer(payload):
        decisions.append(payload)
        return {"action": "reject", "note": "test"}

    insight_graph.run(con, reviewer=reviewer)
    digest.run(con, "TEST")
    con.close()
    yield {"decisions": decisions}
    settings.db_path, settings.reports_dir, settings.llm_backend, settings.embed_backend = old


def test_pipeline_produces_themes_and_tickets(demo_db):
    from voc.db import connect

    con = connect(settings.db_path, read_only=True)
    n_themes = con.execute("SELECT count(*) FROM themes").fetchone()[0]
    statuses = dict(con.execute("SELECT status, count(*) FROM tickets GROUP BY 1").fetchall())
    orphan = con.execute("""SELECT count(*) FROM theme_evidence te LEFT JOIN reviews r USING (review_id)
                            WHERE r.review_id IS NULL""").fetchone()[0]
    con.close()
    assert n_themes > 10
    assert orphan == 0, "every piece of theme evidence must point at a real review"
    # The interrupt fired for P1 themes and our reviewer's rejection was recorded.
    assert demo_db["decisions"], "P1 themes should pause for human review"
    assert statuses.get("rejected", 0) == len(demo_db["decisions"])
    assert statuses.get("auto_created", 0) > 0


def test_safety_flag_does_not_leak_across_aspects(demo_db):
    """Regression: a severity-3 authenticity span must not make the packaging theme P1."""
    from voc.db import connect

    con = connect(settings.db_path, read_only=True)
    bad = con.execute("""
        SELECT count(*) FROM theme_metrics m JOIN themes t USING (theme_id)
        WHERE m.priority = 'P1' AND t.aspect NOT IN ('skin_reaction', 'authenticity')
    """).fetchone()[0]
    con.close()
    assert bad == 0


def test_service_layer(demo_db):
    from voc.serve import service

    products = service.list_products()
    assert any(p["product_id"] == "me_rosemary_shampoo" for p in products)
    top = service.top_themes("aq_dewy_sunscreen")
    assert top and top[0]["polarity"] == "negative"
    detail = service.theme_detail(top[0]["theme_id"])
    assert detail["evidence"] and all(e["quote"] for e in detail["evidence"])
    hits = service.search_reviews("pump leaking", k=5)
    assert any("leak" in h["text"].lower() for h in hits)
    cmp = service.compare_products("aq_dewy_sunscreen", "comp_minimalist_sunscreen")
    assert cmp["by_aspect"]
    assert "error" in service.run_metrics_sql("DROP TABLE reviews")
    ok = service.run_metrics_sql("SELECT count(*) AS n FROM reviews")
    assert ok["rows"][0]["n"] > 0
    assert "Synthetic demo data" in service.get_digest("TEST")["markdown"]


def test_mcp_server_registers_tools(demo_db):
    from voc.serve.mcp_server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {"list_products", "top_themes", "theme_detail", "compare_products", "search_reviews",
            "get_schema", "run_metrics_sql", "list_tickets", "get_digest"} <= names


def test_api(demo_db):
    from fastapi.testclient import TestClient

    from voc.serve.api import app

    c = TestClient(app)
    assert c.get("/health").json() == {"ok": True}
    assert c.get("/themes", params={"product_id": "me_rosemary_shampoo"}).status_code == 200
    assert c.post("/sql", json={"sql": "DELETE FROM tickets"}).status_code == 400
