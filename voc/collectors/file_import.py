"""Import reviews from a CSV or JSONL file (public datasets, or reviews you collected by hand).

Only import data you're allowed to use. Scraping marketplaces at scale usually breaks their terms.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from voc.collectors.common import author_hash, insert_raw, parse_ts, stable_id


def read_records(path: Path) -> list[dict]:
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data["records"]
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def import_file(
    con,
    path: Path,
    source: str,
    text_col: str = "text",
    product_id: str | None = None,
    product_col: str = "product_id",
    date_col: str | None = "created_at",
    url_col: str | None = "source_url",
    author_col: str | None = "author",
    id_col: str | None = None,
    likes_col: str | None = "likes",
) -> int:
    rows = []
    for rec in read_records(Path(path)):
        text = (rec.get(text_col) or "").strip()
        pid = product_id or rec.get(product_col)
        if not text or not pid:
            continue
        rid = rec.get(id_col) if id_col else None
        rows.append({
            "review_id": (rid if str(rid).startswith(f"{source}:") else f"{source}:{rid}") if rid
                         else stable_id(f"{source}:", pid, text),
            "source": source,
            "source_url": rec.get(url_col) if url_col else None,
            "product_id": pid,
            "text": text,
            "author_hash": author_hash(rec.get(author_col)) if author_col else None,
            "created_at": parse_ts(rec.get(date_col)) if date_col else None,
            "likes": int(float(rec.get(likes_col) or 0)) if likes_col and rec.get(likes_col) else 0,
            "parent_context": rec.get("parent_context"),
        })
    return insert_raw(con, rows)
