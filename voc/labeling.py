"""Fast terminal labelling for the gold set. ~150 reviews takes about 40 minutes.

Codes: one letter per aspect + sentiment sign, comma-separated.
  e=efficacy  r=skin_reaction  f=fragrance  t=texture  p=packaging  v=price_value  a=authenticity  d=delivery
  +  positive   -  negative   0  neutral
  Example: "e-,p-" = negative efficacy and negative packaging.
  x = not relevant   (blank) = relevant but no aspect   s = skip   q = quit
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from voc.config import product_map
from voc.db import rows

CODES = {"e": "efficacy", "r": "skin_reaction", "f": "fragrance", "t": "texture", "p": "packaging",
         "v": "price_value", "a": "authenticity", "d": "delivery"}
SIGNS = {"+": "positive", "-": "negative", "0": "neutral"}


def parse(code: str) -> list[dict] | None:
    out = []
    for part in [c.strip() for c in code.split(",") if c.strip()]:
        if len(part) != 2 or part[0] not in CODES or part[1] not in SIGNS:
            return None
        out.append({"aspect": CODES[part[0]], "sentiment": SIGNS[part[1]]})
    return out


def sample(con, n: int, seed: int, exclude: set[str]) -> list[dict]:
    """Stratified by product so every product is represented in the gold set."""
    all_rows = rows(con, "SELECT review_id, product_id, text FROM reviews ORDER BY review_id")
    by_p: dict[str, list] = {}
    for r in all_rows:
        if r["review_id"] not in exclude:
            by_p.setdefault(r["product_id"], []).append(r)
    rng = random.Random(seed)
    for v in by_p.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < n and any(i < len(v) for v in by_p.values()):
        for v in by_p.values():
            if i < len(v) and len(picked) < n:
                picked.append(v[i])
        i += 1
    return picked


def label(con, out_path: Path, n: int = 150, seed: int = 7) -> int:
    done = set()
    if out_path.exists():
        done = {json.loads(line)["review_id"] for line in out_path.read_text().splitlines() if line.strip()}
    products = product_map()
    todo = sample(con, n - len(done), seed, done) if n > len(done) else []
    print(__doc__)
    print(f"{len(done)} already labelled, {len(todo)} to go. Writing to {out_path}\n")
    added = 0
    with out_path.open("a", encoding="utf-8") as f:
        for k, r in enumerate(todo, 1):
            print(f"[{k}/{len(todo)}] {products.get(r['product_id'], {}).get('name', r['product_id'])}")
            print(f"  {r['text']}\n")
            while True:
                code = input("  label> ").strip().lower()
                if code == "q":
                    return added
                if code == "s":
                    break
                if code == "x":
                    rec = {**r, "is_relevant": False, "aspects": []}
                else:
                    aspects = parse(code)
                    if aspects is None:
                        print("  ? couldn't parse, e.g. 'e-,p-' or 'x'")
                        continue
                    rec = {**r, "is_relevant": True, "aspects": aspects}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                added += 1
                break
            print()
    return added
