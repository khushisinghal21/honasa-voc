"""DuckDB storage. One file holds every stage's output, so the whole system is queryable with SQL."""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from voc.config import load_products, settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR PRIMARY KEY,
    brand VARCHAR,
    name VARCHAR,
    category VARCHAR,
    is_competitor BOOLEAN
);

-- Exactly what the collectors fetched. Author identity is only ever stored as a salted hash.
CREATE TABLE IF NOT EXISTS raw_reviews (
    review_id VARCHAR PRIMARY KEY,
    source VARCHAR,
    source_url VARCHAR,
    product_id VARCHAR,
    text VARCHAR,
    author_hash VARCHAR,
    created_at TIMESTAMP,
    likes INTEGER,
    parent_context VARCHAR,
    collected_at TIMESTAMP
);

-- Cleaned, de-duplicated, PII-scrubbed reviews. This is what every later stage reads.
CREATE TABLE IF NOT EXISTS reviews (
    review_id VARCHAR PRIMARY KEY,
    product_id VARCHAR,
    source VARCHAR,
    source_url VARCHAR,
    text VARCHAR,
    lang VARCHAR,
    created_at TIMESTAMP,
    likes INTEGER
);

CREATE TABLE IF NOT EXISTS extractions (
    review_id VARCHAR PRIMARY KEY,
    product_id VARCHAR,
    is_relevant BOOLEAN,
    overall_sentiment VARCHAR,
    backend VARCHAR,
    model VARCHAR,
    extracted_at TIMESTAMP
);

-- One row per aspect mention. evidence_span is guaranteed to be verbatim text from the review.
CREATE TABLE IF NOT EXISTS aspects (
    review_id VARCHAR,
    product_id VARCHAR,
    aspect VARCHAR,
    sentiment VARCHAR,
    severity INTEGER,
    evidence_span VARCHAR,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS themes (
    theme_id VARCHAR PRIMARY KEY,
    run_id VARCHAR,
    product_id VARCHAR,
    aspect VARCHAR,
    polarity VARCHAR,
    label VARCHAR,
    summary VARCHAR,
    n_reviews INTEGER,
    n_mentions INTEGER,
    cohesion DOUBLE,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS theme_evidence (
    theme_id VARCHAR,
    review_id VARCHAR,
    evidence_span VARCHAR,
    rank INTEGER
);

CREATE TABLE IF NOT EXISTS theme_metrics (
    theme_id VARCHAR PRIMARY KEY,
    n_reviews INTEGER,
    neg_share DOUBLE,
    share_of_aspect DOUBLE,
    recent_count INTEGER,
    prior_count INTEGER,
    trend DOUBLE,
    safety_flag BOOLEAN,
    severity_score DOUBLE,
    priority VARCHAR,
    team VARCHAR,
    routing_rationale VARCHAR
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id VARCHAR PRIMARY KEY,
    theme_id VARCHAR,
    product_id VARCHAR,
    theme_label VARCHAR,
    team VARCHAR,
    priority VARCHAR,
    severity_score DOUBLE,
    status VARCHAR,          -- approved | rejected | auto_created | pending_review
    reviewer_note VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS digests (
    period VARCHAR PRIMARY KEY,
    markdown VARCHAR,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR,
    step VARCHAR,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    stats VARCHAR
);
"""

# Tables the read-only SQL tool may touch. raw_reviews is deliberately excluded (it holds author hashes).
QUERYABLE_TABLES = [
    "products", "reviews", "extractions", "aspects", "themes",
    "theme_evidence", "theme_metrics", "tickets",
]


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def connect(path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        return duckdb.connect(str(path), read_only=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con


def sync_products(con: duckdb.DuckDBPyConnection, products: list[dict] | None = None) -> None:
    products = products or load_products()
    con.execute("DELETE FROM products")
    con.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
        [(p["id"], p["brand"], p["name"], p["category"], bool(p.get("is_competitor", False))) for p in products],
    )


@contextmanager
def run_step(con: duckdb.DuckDBPyConnection, step: str):
    """Log a pipeline step with its stats so you can show run history in the README."""
    run_id, started, stats = new_id("run_"), now(), {}
    yield run_id, stats
    con.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
        [run_id, step, started, now(), json.dumps(stats, default=str)],
    )


def rows(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> list[dict]:
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
