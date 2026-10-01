from __future__ import annotations

import hashlib
from datetime import datetime

from voc.config import settings
from voc.db import now


def author_hash(author: str | None) -> str | None:
    """Never store usernames or channel IDs. A salted hash still lets us count repeat posters."""
    if not author:
        return None
    return hashlib.sha256((settings.author_salt + author).encode()).hexdigest()[:16]


def stable_id(prefix: str, *parts: str) -> str:
    return prefix + hashlib.sha1("||".join(parts).encode()).hexdigest()[:16]


def parse_ts(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(value)
    s = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(s[:10], fmt)
            except ValueError:
                continue
    return None


def insert_raw(con, rows: list[dict]) -> int:
    """Insert collector rows; duplicates by review_id are ignored. Returns rows actually added."""
    if not rows:
        return 0
    before = con.execute("SELECT count(*) FROM raw_reviews").fetchone()[0]
    con.executemany(
        "INSERT OR IGNORE INTO raw_reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(
            r["review_id"], r["source"], r.get("source_url"), r["product_id"], r["text"],
            r.get("author_hash"), r.get("created_at"), int(r.get("likes") or 0),
            r.get("parent_context"), now(),
        ) for r in rows],
    )
    return con.execute("SELECT count(*) FROM raw_reviews").fetchone()[0] - before


def mentions_product(text: str, product: dict) -> bool:
    t = (text or "").lower()
    return any(term in t for term in product.get("match_terms", []))
