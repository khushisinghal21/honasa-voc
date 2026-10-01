"""`voc` command-line interface. Run `voc --help` for everything."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import typer

from voc.config import ROOT, load_products, settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Honasa Voice of Consumer POC")
collect_app = typer.Typer(no_args_is_help=True, help="Collect raw comments")
eval_app = typer.Typer(no_args_is_help=True, help="Run evaluations")
serve_app = typer.Typer(no_args_is_help=True, help="Serve the data")
app.add_typer(collect_app, name="collect")
app.add_typer(eval_app, name="eval")
app.add_typer(serve_app, name="serve")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "anthropic", "urllib3", "prawcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _show(title: str, stats: dict) -> None:
    typer.secho(f"\n== {title}", fg="cyan", bold=True)
    typer.echo(json.dumps(stats, indent=2, default=str))


def _con():
    from voc.db import connect, sync_products

    con = connect()
    sync_products(con)
    return con


def _products(only: list[str] | None) -> list[dict]:
    ps = load_products()
    return [p for p in ps if not only or p["id"] in only]


# ----------------------------------------------------------------- setup + collection

@app.command()
def init():
    """Create the database and load products.yaml."""
    con = _con()
    _show("init", {"db": str(settings.db_path), "products": len(load_products())})
    con.close()


@collect_app.command("youtube")
def collect_youtube(product: list[str] = typer.Option(None, help="Limit to product ids"),
                    videos_per_query: int = 4, max_comments: int = 500):
    """Comments from review videos (YouTube Data API v3)."""
    from voc.collectors import youtube

    con = _con()
    _show("youtube", youtube.collect(con, _products(product), videos_per_query, max_comments))


@collect_app.command("reddit")
def collect_reddit(product: list[str] = typer.Option(None, help="Limit to product ids"),
                   posts_per_query: int = 25, time_filter: str = "year"):
    """Posts and comments from configured subreddits (Reddit API via PRAW)."""
    from voc.collectors import reddit

    con = _con()
    _show("reddit", reddit.collect(con, _products(product), posts_per_query, time_filter))


@collect_app.command("file")
def collect_file(path: Path, source: str = typer.Option(..., help="Label for where this data came from"),
                 text_col: str = "text", product_id: str = typer.Option(None, help="Use one product for all rows"),
                 product_col: str = "product_id", date_col: str = "created_at", url_col: str = "source_url"):
    """Import a CSV/JSONL file of reviews you're allowed to use."""
    from voc.collectors.file_import import import_file

    con = _con()
    n = import_file(con, path, source, text_col, product_id, product_col, date_col, url_col)
    _show("import", {"path": str(path), "added": n})


# ----------------------------------------------------------------- pipeline

@app.command()
def clean():
    """Scrub PII, tag language, drop junk + duplicates."""
    from voc.pipeline import clean as m

    _show("clean", m.run(_con()))


@app.command()
def extract(limit: int = typer.Option(None, help="Only process N reviews (try 50 first to check cost)")):
    """Aspect-level sentiment with verbatim-evidence verification."""
    from voc.pipeline import extract as m

    _show("extract", m.run(_con(), limit))


@app.command()
def themes():
    """Cluster evidence into labelled themes."""
    from voc.pipeline import themes as m

    _show("themes", m.run(_con()))


def _interactive_reviewer(payload: dict) -> dict:
    typer.secho(f"\n⚠  {payload.get('priority')} theme needs approval", fg="yellow", bold=True)
    typer.echo(f"   Product : {payload.get('product_id')}")
    typer.echo(f"   Theme   : {payload.get('label') or payload.get('theme_label')}")
    typer.echo(f"   Summary : {payload.get('summary')}")
    typer.echo(f"   Owner   : {payload.get('team')}   score={payload.get('severity_score')}")
    choice = typer.prompt("   [a]pprove / [r]eject / [d]efer", default="a").strip().lower()[:1]
    action = {"a": "approve", "r": "reject"}.get(choice, "defer")
    note = typer.prompt("   note (optional)", default="", show_default=False) if action != "defer" else None
    return {"action": action, "note": note or None}


@app.command()
def insights(auto_approve: bool = typer.Option(False, help="Approve P1 tickets without asking (demo only)"),
             no_input: bool = typer.Option(False, help="Don't prompt; P1 tickets wait as pending_review")):
    """Score, route and raise tickets (LangGraph; P1 pauses for human approval)."""
    from voc.graph import insight_graph as m

    reviewer = None if (auto_approve or no_input) else _interactive_reviewer
    _show("insights", m.run(_con(), reviewer=reviewer, auto_approve=auto_approve))


@app.command()
def review():
    """Approve or reject tickets left as pending_review."""
    from voc.graph.insight_graph import resolve_pending

    n = resolve_pending(_con(), _interactive_reviewer)
    _show("review", {"pending_seen": n})


@app.command()
def digest(period: str = typer.Option(None, help="Label, e.g. 2026-W40 (default: this ISO week)")):
    """Write the weekly Markdown digest to reports/."""
    from voc.pipeline import digest as m

    _show("digest", m.run(_con(), period))


@app.command("run")
def run_all(auto_approve: bool = False, no_input: bool = False, limit: int = typer.Option(None)):
    """clean → extract → themes → insights → digest."""
    clean()
    extract(limit)
    themes()
    insights(auto_approve, no_input)
    digest(None)


@app.command()
def stats():
    """Row counts per table and recent runs."""
    from voc.db import rows

    con = _con()
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("raw_reviews", "reviews", "extractions", "aspects", "themes", "tickets")}
    _show("tables", counts)
    _show("recent runs", {"runs": rows(con, "SELECT step, finished_at, stats FROM runs ORDER BY finished_at DESC LIMIT 8")})


@app.command()
def demo():
    """Full offline run on SYNTHETIC data (no API keys): baseline extractor + TF-IDF embeddings."""
    from voc.collectors.file_import import import_file
    from voc.db import connect, sync_products

    settings.db_path = ROOT / "data/demo.duckdb"
    settings.reports_dir = ROOT / "reports/demo"
    settings.llm_backend = "baseline"
    settings.embed_backend = "tfidf"
    if settings.db_path.exists():
        settings.db_path.unlink()
    sample = ROOT / "data/sample/synthetic_reviews.jsonl"
    if not sample.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic.py")], check=True)
    con = connect()
    sync_products(con)
    _show("import (synthetic)", {"added": import_file(con, sample, "synthetic", url_col=None, id_col="review_id")})
    con.close()
    run_all(auto_approve=True, no_input=True, limit=None)
    eval_extraction(gold=ROOT / "evals/demo_gold_synthetic.jsonl", backend="baseline")
    eval_sql_cmd()
    typer.secho(f"\nDemo done. DB: {settings.db_path}  Reports: {settings.reports_dir}", fg="green", bold=True)
    typer.secho("Reminder: synthetic data. Don't quote these numbers anywhere.", fg="yellow")


# ----------------------------------------------------------------- labelling + evals

@app.command()
def label(n: int = 150, out: Path = typer.Option(None, help="Defaults to evals/gold_labels.jsonl")):
    """Hand-label a stratified sample for the gold set."""
    from voc.evaluation import DEFAULT_GOLD
    from voc.labeling import label as run_label

    added = run_label(_con(), out or DEFAULT_GOLD, n)
    _show("label", {"added": added})


@eval_app.command("extraction")
def eval_extraction(gold: Path = typer.Option(None, help="Defaults to evals/gold_labels.jsonl"),
                    backend: str = typer.Option(None, help="anthropic | baseline (default: VOC_LLM_BACKEND)")):
    """Aspect + sentiment accuracy against the gold set. Run once per backend to compare."""
    from voc.evaluation import DEFAULT_GOLD, eval_extraction as run_eval

    gold = gold or DEFAULT_GOLD
    if not gold.exists():
        raise typer.BadParameter(f"{gold} not found. Create it with `voc label`.")
    res = run_eval(gold, backend or settings.llm_backend)
    _show("eval extraction", {k: v for k, v in res.items() if k not in ("per_aspect", "sample_errors")})


@eval_app.command("themes")
def eval_themes_cmd(limit: int = 40):
    """LLM-as-judge faithfulness of theme labels + cluster cohesion."""
    from voc.evaluation import eval_themes

    _show("eval themes", eval_themes(limit))


@eval_app.command("sql")
def eval_sql_cmd():
    """SQL guard red-team + NL→SQL execution accuracy."""
    from voc.evaluation import SQL_QUESTIONS, eval_sql

    res = eval_sql(SQL_QUESTIONS)
    _show("eval sql", {k: v for k, v in res.items() if k != "guard_details"})


# ----------------------------------------------------------------- serving

@serve_app.command("mcp")
def serve_mcp(transport: str = typer.Option("stdio", help="stdio | http"), host: str = "127.0.0.1", port: int = 8000):
    """MCP server for Claude Desktop (stdio) or remote clients (http)."""
    logging.getLogger().setLevel(logging.WARNING)  # stdout belongs to the MCP protocol in stdio mode
    from voc.serve.mcp_server import main

    main("stdio" if transport == "stdio" else "streamable-http", host, port)


@serve_app.command("api")
def serve_api(host: str = "127.0.0.1", port: int = 8080):
    """HTTP API (FastAPI)."""
    import uvicorn

    uvicorn.run("voc.serve.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
