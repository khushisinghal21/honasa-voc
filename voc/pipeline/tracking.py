"""Stage 3b: track themes across runs, so the same consumer issue keeps one identity week to week.

Themes are rebuilt from scratch on every run (new data can split or merge clusters), so theme_ids
change every time. A brand team doesn't care about that: they want to know whether the pump complaint
is *new*, *growing*, *steady*, *shrinking* or has *faded*. This module links each new theme to the
track it continues.

Matching happens only inside the same (product, aspect, polarity) group, using two signals:
  - review overlap: share of the smaller theme's reviews that also back the other one. Old reviews
    stay in the database, so a continuing issue keeps most of its evidence.
  - centroid similarity: old and new evidence spans embedded *together* (so it works with the
    TF-IDF fallback too), then the cosine between the two themes' centroids. This catches an issue
    that continues with mostly new reviews.
Pairs are matched greedily, best score first, one to one.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np

from voc.db import new_id, now, rows
from voc.embeddings import embed

log = logging.getLogger(__name__)

MIN_OVERLAP = 0.4      # share of the smaller theme's reviews that must be shared
MIN_COSINE = 0.75      # or: centroid similarity of the evidence spans
GROWTH_MIN_ABS = 3     # a change smaller than this is noise, whatever the percentage
GROWTH_MIN_REL = 0.25


def classify(prev_n: int, cur_n: int) -> str:
    diff = cur_n - prev_n
    if diff >= GROWTH_MIN_ABS and diff >= GROWTH_MIN_REL * max(prev_n, 1):
        return "growing"
    if -diff >= GROWTH_MIN_ABS and -diff >= GROWTH_MIN_REL * max(prev_n, 1):
        return "shrinking"
    return "steady"


def overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def match(prev: list[dict], cur: list[dict]) -> list[tuple[int, int, float, float]]:
    """prev/cur items: {"review_ids": set, "spans": list[str]}. Returns (i_prev, j_cur, overlap, cosine)."""
    if not prev or not cur:
        return []
    spans = [s for p in prev for s in p["spans"]] + [s for c in cur for s in c["spans"]]
    vecs = embed(spans)

    def centroids(items, start):
        out, pos = [], start
        for it in items:
            v = vecs[pos:pos + len(it["spans"])].mean(axis=0)
            out.append(v / (np.linalg.norm(v) or 1.0))
            pos += len(it["spans"])
        return out, pos

    cp, end = centroids(prev, 0)
    cc, _ = centroids(cur, end)
    cands = []
    for i, p in enumerate(prev):
        for j, c in enumerate(cur):
            ov = overlap(p["review_ids"], c["review_ids"])
            cos = float(cp[i] @ cc[j])
            if ov >= MIN_OVERLAP or cos >= MIN_COSINE:
                cands.append((max(ov, cos), i, j, ov, cos))
    used_p, used_c, out = set(), set(), []
    for _, i, j, ov, cos in sorted(cands, reverse=True):
        if i not in used_p and j not in used_c:
            used_p.add(i)
            used_c.add(j)
            out.append((i, j, round(ov, 3), round(cos, 3)))
    return out


def _current_themes(con) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in rows(con, "SELECT theme_id, product_id, aspect, polarity, label, n_reviews FROM themes"):
        ev = con.execute("SELECT review_id, evidence_span FROM theme_evidence WHERE theme_id = ? ORDER BY rank",
                         [t["theme_id"]]).fetchall()
        t["review_ids"] = {r for r, _ in ev}
        t["spans"] = [s for _, s in ev]
        groups[(t["product_id"], t["aspect"], t["polarity"])].append(t)
    return groups


def _previous_snapshots(con) -> dict[tuple, list[dict]]:
    """Latest snapshot of every track that was alive at the end of the previous run."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in rows(con, """
        SELECT tr.track_id, tr.product_id, tr.aspect, tr.polarity, s.n_reviews, s.review_ids, s.spans
        FROM theme_tracks tr
        JOIN theme_track_runs s ON s.track_id = tr.track_id AND s.run_id = tr.last_run_id
        WHERE tr.status <> 'faded'
    """):
        t["review_ids"] = set(json.loads(t["review_ids"]))
        t["spans"] = json.loads(t["spans"])
        groups[(t["product_id"], t["aspect"], t["polarity"])].append(t)
    return groups


def link(con, run_id: str) -> dict:
    """Assign every theme of this run to a track. Call right after themes are rebuilt."""
    cur_groups, prev_groups = _current_themes(con), _previous_snapshots(con)
    first_run = not any(prev_groups.values()) and con.execute(
        "SELECT count(*) FROM theme_tracks").fetchone()[0] == 0
    stats = {"new": 0, "growing": 0, "steady": 0, "shrinking": 0, "faded": 0, "first_run": first_run}
    ts = now()

    for key in sorted(set(cur_groups) | set(prev_groups)):
        cur, prev = cur_groups.get(key, []), prev_groups.get(key, [])
        pairs = match(prev, cur)
        matched_p = {i for i, *_ in pairs}
        matched_c = {j for _, j, *_ in pairs}

        for i, j, ov, cos in pairs:
            p, c = prev[i], cur[j]
            status = classify(p["n_reviews"], c["n_reviews"])
            con.execute("""
                UPDATE theme_tracks SET label = ?, last_seen = ?, last_run_id = ?, runs_seen = runs_seen + 1,
                       status = ?, prev_n_reviews = ?, n_reviews = ?, match_overlap = ?, match_cosine = ?
                WHERE track_id = ?
            """, [c["label"], ts, run_id, status, p["n_reviews"], c["n_reviews"], ov, cos, p["track_id"]])
            _snapshot(con, p["track_id"], run_id, c, ts)
            stats[status] += 1

        for j, c in enumerate(cur):
            if j in matched_c:
                continue
            tid = new_id("tr_")
            con.execute("INSERT INTO theme_tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, NULL, ?, NULL, NULL)", [
                tid, key[0], key[1], key[2], c["label"], ts, ts, run_id, run_id,
                "new", c["n_reviews"],
            ])
            _snapshot(con, tid, run_id, c, ts)
            stats["new"] += 1

        for i, p in enumerate(prev):
            if i not in matched_p:
                con.execute("UPDATE theme_tracks SET status = 'faded', prev_n_reviews = n_reviews, n_reviews = 0 "
                            "WHERE track_id = ?", [p["track_id"]])
                stats["faded"] += 1
    return stats


def _snapshot(con, track_id: str, run_id: str, theme: dict, ts) -> None:
    con.execute("INSERT INTO theme_track_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        track_id, run_id, theme["theme_id"], theme["label"], theme["n_reviews"],
        json.dumps(sorted(theme["review_ids"])), json.dumps(theme["spans"], ensure_ascii=False), ts,
    ])


def track_of(con, theme_id: str) -> dict | None:
    r = rows(con, """
        SELECT tr.* FROM theme_track_runs s JOIN theme_tracks tr USING (track_id) WHERE s.theme_id = ?
    """, [theme_id])
    return r[0] if r else None
