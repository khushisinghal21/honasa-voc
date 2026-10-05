"""The query layer. Plain functions so the MCP server, the HTTP API, and the tests all share one implementation.
Every function opens a short-lived read-only connection."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from voc.config import settings
from voc.db import QUERYABLE_TABLES, connect, rows
from voc.retrieval.hybrid import HybridSearcher
from voc.serve.sql_guard import UnsafeSQL, validate


def _ro():
    return connect(settings.db_path, read_only=True)


def _jsonable(rs: list[dict]) -> list[dict]:
    return [{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in r.items()} for r in rs]


def list_products() -> list[dict]:
    with _ro() as con:
        return _jsonable(rows(con, """
            SELECT p.*, count(DISTINCT r.review_id) AS reviews,
                   count(DISTINCT e.review_id) FILTER (WHERE e.is_relevant) AS relevant_reviews
            FROM products p
            LEFT JOIN reviews r USING (product_id)
            LEFT JOIN extractions e USING (review_id)
            GROUP BY ALL ORDER BY p.is_competitor, p.name
        """))


def top_themes(product_id: str | None = None, aspect: str | None = None,
               polarity: str = "negative", limit: int = 10) -> list[dict]:
    where, params = ["t.polarity = ?"], [polarity]
    if product_id:
        where.append("t.product_id = ?")
        params.append(product_id)
    if aspect:
        where.append("t.aspect = ?")
        params.append(aspect)
    with _ro() as con:
        return _jsonable(rows(con, f"""
            SELECT t.theme_id, t.product_id, t.aspect, t.polarity, t.label, t.summary, t.n_reviews,
                   m.share_of_aspect, m.recent_count, m.prior_count, m.priority, m.severity_score, m.team
            FROM themes t LEFT JOIN theme_metrics m USING (theme_id)
            WHERE {' AND '.join(where)}
            ORDER BY m.severity_score DESC NULLS LAST, t.n_reviews DESC
            LIMIT ?
        """, params + [int(limit)]))


def theme_detail(theme_id: str, max_quotes: int = 15) -> dict:
    with _ro() as con:
        theme = rows(con, """
            SELECT t.*, m.* EXCLUDE (theme_id, n_reviews) FROM themes t
            LEFT JOIN theme_metrics m USING (theme_id) WHERE t.theme_id = ?
        """, [theme_id])
        if not theme:
            return {"error": f"no theme {theme_id}"}
        quotes = rows(con, """
            SELECT te.evidence_span AS quote, r.review_id, r.source, r.source_url, r.lang, r.created_at, r.likes
            FROM theme_evidence te JOIN reviews r USING (review_id)
            WHERE te.theme_id = ? ORDER BY te.rank LIMIT ?
        """, [theme_id, int(max_quotes)])
        tickets = rows(con, "SELECT ticket_id, team, priority, status, reviewer_note FROM tickets WHERE theme_id = ?",
                       [theme_id])
        return {"theme": _jsonable(theme)[0], "evidence": _jsonable(quotes), "tickets": _jsonable(tickets)}


def theme_changes(product_id: str | None = None, polarity: str = "negative") -> dict:
    """How each consumer issue moved since the previous run: new, growing, steady, shrinking or faded."""
    where, params = ["tr.polarity = ?"], [polarity]
    if product_id:
        where.append("tr.product_id = ?")
        params.append(product_id)
    with _ro() as con:
        runs = con.execute("SELECT count(DISTINCT run_id) FROM theme_track_runs").fetchone()[0]
        data = rows(con, f"""
            SELECT tr.track_id, tr.product_id, tr.aspect, tr.label, tr.status, tr.prev_n_reviews, tr.n_reviews,
                   tr.runs_seen, tr.first_seen, tr.last_seen, s.theme_id AS current_theme_id
            FROM theme_tracks tr
            LEFT JOIN theme_track_runs s ON s.track_id = tr.track_id AND s.run_id = tr.last_run_id
                                       AND tr.status <> 'faded'
            WHERE {' AND '.join(where)}
            ORDER BY CASE tr.status WHEN 'new' THEN 0 WHEN 'growing' THEN 1 WHEN 'faded' THEN 2
                                    WHEN 'shrinking' THEN 3 ELSE 4 END, tr.n_reviews DESC
        """, params)
    by_status: dict[str, list] = {}
    for r in _jsonable(data):
        by_status.setdefault(r["status"], []).append(r)
    note = ("Only one run so far, so every theme is 'new'. Run the pipeline again on fresh data to see change."
            if runs < 2 else "Compared with the previous run. Use current_theme_id with theme_detail for quotes.")
    return {"runs_compared": runs, "summary": {k: len(v) for k, v in by_status.items()},
            "by_status": by_status, "note": note}


def compare_products(product_a: str, product_b: str) -> dict:
    """Aspect-level sentiment for two products side by side, normalised per 100 relevant reviews."""
    with _ro() as con:
        data = rows(con, """
            WITH base AS (
                SELECT product_id, count(*) AS relevant FROM extractions
                WHERE is_relevant AND product_id IN (?, ?) GROUP BY 1
            )
            SELECT a.product_id, a.aspect,
                   count(*) FILTER (WHERE a.sentiment = 'positive') AS positive,
                   count(*) FILTER (WHERE a.sentiment = 'negative') AS negative,
                   round(100.0 * count(*) FILTER (WHERE a.sentiment = 'negative') / any_value(b.relevant), 1)
                       AS negative_per_100_reviews,
                   any_value(b.relevant) AS relevant_reviews
            FROM aspects a JOIN extractions e USING (review_id) JOIN base b ON b.product_id = a.product_id
            WHERE e.is_relevant
            GROUP BY 1, 2 ORDER BY 2, 1
        """, [product_a, product_b])
    return {"products": [product_a, product_b], "by_aspect": data,
            "note": "Normalised per 100 relevant reviews so products with different volumes are comparable."}


@lru_cache(maxsize=1)
def _searcher(db_mtime: float) -> HybridSearcher:
    with _ro() as con:
        docs = rows(con, """
            SELECT r.review_id, r.product_id, r.source, r.source_url, r.text, r.lang, r.created_at
            FROM reviews r JOIN extractions e USING (review_id) WHERE e.is_relevant
        """)
    return HybridSearcher(_jsonable(docs))


def search_reviews(query: str, product_id: str | None = None, k: int = 10) -> list[dict]:
    return _searcher(settings.db_path.stat().st_mtime).search(query, k=k, product_id=product_id)


def get_schema() -> dict:
    """Column names and types for every queryable table, so a model can write correct SQL."""
    with _ro() as con:
        out = {}
        for t in QUERYABLE_TABLES:
            out[t] = [{"column": r[0], "type": r[1]} for r in con.execute(f"DESCRIBE {t}").fetchall()]
    out["_notes"] = (
        "aspects.sentiment ∈ positive|negative|neutral. aspects.aspect ∈ efficacy, skin_reaction, fragrance, "
        "texture, packaging, price_value, authenticity, delivery. Always join extractions and filter "
        "is_relevant = true when counting consumer opinions. themes.polarity ∈ positive|negative. "
        "theme_metrics.priority ∈ P1|P2|P3. reviews.lang ∈ en|hi|hinglish. "
        "theme_tracks is one row per consumer issue across runs; theme_tracks.status ∈ "
        "new|growing|steady|shrinking|faded."
    )
    return out


def run_metrics_sql(sql: str) -> dict:
    try:
        safe = validate(sql)
    except UnsafeSQL as e:
        return {"error": str(e)}
    with _ro() as con:
        try:
            cur = con.execute(safe)
        except Exception as e:
            return {"error": f"query failed: {e}"}
        cols = [d[0] for d in cur.description]
        data = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"columns": cols, "rows": _jsonable(data), "row_count": len(data), "executed_sql": safe}


def list_tickets(status: str | None = None) -> list[dict]:
    with _ro() as con:
        if status:
            return _jsonable(rows(con, "SELECT * FROM tickets WHERE status = ? ORDER BY priority, severity_score DESC",
                                  [status]))
        return _jsonable(rows(con, "SELECT * FROM tickets ORDER BY priority, severity_score DESC"))


def get_digest(period: str | None = None) -> dict:
    with _ro() as con:
        if period:
            r = rows(con, "SELECT * FROM digests WHERE period = ?", [period])
        else:
            r = rows(con, "SELECT * FROM digests ORDER BY created_at DESC LIMIT 1")
    return _jsonable(r)[0] if r else {"error": "no digest yet; run `voc digest`"}
