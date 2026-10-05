"""Theme tracking across runs, and human decisions that stick to an issue instead of a run."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

from voc.config import ROOT, settings
from voc.pipeline import tracking


# ---------------------------------------------------------------- matching logic

def test_classify_ignores_small_wobbles():
    assert tracking.classify(10, 11) == "steady"       # +1 is noise
    assert tracking.classify(10, 14) == "growing"      # +4 and +40%
    assert tracking.classify(40, 43) == "steady"       # +3 but only +7.5%
    assert tracking.classify(12, 6) == "shrinking"


def test_match_pairs_by_shared_reviews_one_to_one():
    prev = [{"review_ids": {"a", "b", "c", "d"}, "spans": ["pump leaks in my bag", "pump leaks everywhere"]},
            {"review_ids": {"x", "y", "z"}, "spans": ["smells like rosemary", "lovely fresh smell"]}]
    cur = [{"review_ids": {"x", "y", "z", "w"}, "spans": ["smells like rosemary", "fresh smell"]},
           {"review_ids": {"a", "b", "c", "e", "f"}, "spans": ["pump leaks in my bag", "pump broke"]}]
    pairs = {(i, j) for i, j, *_ in tracking.match(prev, cur)}
    assert pairs == {(0, 1), (1, 0)}


def test_match_leaves_unrelated_themes_unmatched():
    prev = [{"review_ids": {"a", "b", "c"}, "spans": ["the pump stopped working halfway"]}]
    cur = [{"review_ids": {"p", "q", "r"}, "spans": ["got an allergic rash on my cheeks"]}]
    assert tracking.match(prev, cur) == []


# ---------------------------------------------------------------- two runs, two weeks apart

def _run_twice(tmp, reviewer, classify_override=None):
    """Run 1 on reviews older than 14 days, run 2 on everything. Returns the reviewer payloads per run."""
    from voc.collectors.file_import import import_file
    from voc.db import connect, sync_products
    from voc.graph import insight_graph
    from voc.pipeline import clean, digest, extract, themes

    settings.db_path, settings.reports_dir = tmp / "t.duckdb", tmp / "reports"
    settings.llm_backend, settings.embed_backend = "baseline", "tfidf"
    sample = ROOT / "data/sample/synthetic_reviews.jsonl"
    if not sample.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic.py")], check=True)
    recs = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines() if line.strip()]
    cutoff = max(datetime.fromisoformat(r["created_at"]) for r in recs) - timedelta(days=14)
    early = tmp / "early.jsonl"
    early.write_text("\n".join(json.dumps(r) for r in recs if datetime.fromisoformat(r["created_at"]) <= cutoff))

    con = connect()
    sync_products(con)
    asked = []
    for n, path in enumerate([early, sample], start=1):
        if n == 2 and classify_override:
            tracking.classify = classify_override
        import_file(con, path, "synthetic", url_col=None, id_col="review_id")
        clean.run(con)
        extract.run(con)
        themes.run(con)
        calls = []

        def review(payload, calls=calls):
            calls.append(payload)
            return reviewer(payload)

        insight_graph.run(con, reviewer=review)
        digest.run(con, f"RUN{n}")
        asked.append(calls)
    con.close()
    return asked


@pytest.fixture
def isolated(tmp_path):
    old = (settings.db_path, settings.reports_dir, settings.llm_backend, settings.embed_backend, tracking.classify)
    yield tmp_path
    settings.db_path, settings.reports_dir, settings.llm_backend, settings.embed_backend, tracking.classify = old


def test_issues_keep_identity_and_decisions_carry_forward(isolated):
    from voc.db import connect
    from voc.serve import service

    asked = _run_twice(isolated, lambda p: {"action": "reject", "note": "seen it, known batch issue"})
    con = connect(settings.db_path, read_only=True)
    status = dict(con.execute("SELECT status, count(*) FROM theme_tracks GROUP BY 1").fetchall())
    continued = con.execute("SELECT count(*) FROM theme_tracks WHERE runs_seen = 2").fetchone()[0]
    per_track = con.execute("""
        SELECT max(c) FROM (SELECT s.track_id, count(*) AS c FROM tickets tk
                            JOIN theme_track_runs s ON s.theme_id = tk.theme_id GROUP BY 1)""").fetchone()[0]
    orphans = con.execute("""SELECT count(*) FROM tickets tk LEFT JOIN themes t USING (theme_id)
                             WHERE t.theme_id IS NULL""").fetchone()[0]
    notes = [r[0] for r in con.execute("SELECT reviewer_note FROM tickets WHERE status = 'rejected'").fetchall()]
    con.close()

    assert continued > 10, "most issues should be recognised as the same issue in run 2"
    assert status.get("steady", 0) > 0
    assert asked[0], "run 1 should pause P1 themes for review"
    # Every run-1 decision was on a non-growing issue in this data, so run 2 must not ask about them again.
    asked_again = {p["label"] for p in asked[1] if "previous_decision" in p}
    assert not asked_again
    assert len(asked[1]) < len(asked[0]) + 2
    assert per_track == 1, "one ticket per issue, updated in place"
    assert orphans == 0, "tickets must follow their issue to the current theme"
    assert any("Original note: seen it" in n for n in notes if n.startswith("Carried forward"))

    run2 = service.get_digest("RUN2")["markdown"]
    assert "What changed since the last run" in run2
    assert "What changed" not in service.get_digest("RUN1")["markdown"]
    changes = service.theme_changes()
    assert changes["runs_compared"] == 2 and changes["summary"]


def test_growing_issue_is_sent_back_for_review(isolated):
    """If an issue a person already decided on gets worse, the old decision expires."""
    asked = _run_twice(isolated, lambda p: {"action": "approve", "note": "ok"},
                       classify_override=lambda prev_n, cur_n: "growing")
    assert asked[0]
    re_asked = [p for p in asked[1] if "previous_decision" in p]
    assert re_asked, "growing P1 issues with an earlier decision must be asked again"
    prev = re_asked[0]["previous_decision"]
    assert prev["status"] == "approved" and prev["reviews_now"] is not None


def test_mcp_exposes_theme_changes():
    import asyncio

    from voc.serve.mcp_server import mcp

    assert "theme_changes" in {t.name for t in asyncio.run(mcp.list_tools())}
