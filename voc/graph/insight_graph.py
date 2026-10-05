"""Stage 4: turn themes into actions with a LangGraph workflow.

    compute_metrics -> score -> route -> save_metrics -> { P1: human_review -> create_ticket
                                                          P2: create_ticket (auto)
                                                          P3: END (logged in digest only) }

P1 themes (safety/trust issues) pause at `human_review` using LangGraph's interrupt(): nothing is
filed until a person approves it. That mirrors the human-in-the-loop approvals the role describes.

Decisions stick to the *issue*, not the run. Theme ids change every run, so each theme is looked up
on its track (pipeline/tracking.py). If a person already approved or rejected that issue, the decision
is carried forward and the existing ticket is updated in place, instead of asking again every week.
The one exception: if the issue is growing, the old decision is out of date and a person is asked again.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, TypedDict

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from voc.db import new_id, now, rows, run_step
from voc.llm import get_llm, use_llm
from voc.pipeline.tracking import track_of
from voc.schemas import RoutingDecision

log = logging.getLogger(__name__)
RULES = yaml.safe_load((Path(__file__).parent / "routing_rules.yaml").read_text())

W_VOLUME, W_NEG, W_SAFETY, W_TREND = 0.35, 0.25, 0.25, 0.15
P2_THRESHOLD = 0.45
TREND_WINDOW_DAYS = 28


class InsightState(TypedDict, total=False):
    theme: dict
    metrics: dict
    severity_score: float
    priority: str
    team: str
    rationale: str
    decision: dict
    ticket_id: str
    status: str
    track: dict            # stable identity of this issue across runs
    prior: dict            # an earlier human decision on the same track, if any
    carried_forward: bool


def build_graph(con):
    def compute_metrics(state: InsightState) -> dict:
        t = state["theme"]
        pid, aspect = t["product_id"], t["aspect"]
        agg = con.execute("""
            SELECT count(DISTINCT a.review_id),
                   count(*) FILTER (WHERE a.sentiment = 'negative'),
                   count(*)
            FROM aspects a JOIN extractions e USING (review_id)
            WHERE e.is_relevant AND a.product_id = ? AND a.aspect = ?
        """, [pid, aspect]).fetchone()
        aspect_reviews, neg_mentions, all_mentions = agg
        max_n = con.execute("SELECT max(n_reviews) FROM themes WHERE product_id = ?", [pid]).fetchone()[0] or 1
        ref = con.execute("SELECT max(created_at) FROM reviews WHERE product_id = ?", [pid]).fetchone()[0]
        recent = prior = 0
        if ref is not None:
            w = timedelta(days=TREND_WINDOW_DAYS)
            recent, prior = con.execute("""
                SELECT count(DISTINCT r.review_id) FILTER (WHERE r.created_at > ?),
                       count(DISTINCT r.review_id) FILTER (WHERE r.created_at <= ? AND r.created_at > ?)
                FROM theme_evidence te JOIN reviews r USING (review_id)
                WHERE te.theme_id = ?
            """, [ref - w, ref - w, ref - 2 * w, t["theme_id"]]).fetchone()
        safety = con.execute("""
            SELECT count(*) > 0 FROM theme_evidence te
            JOIN themes t ON t.theme_id = te.theme_id
            JOIN aspects a ON a.review_id = te.review_id AND a.evidence_span = te.evidence_span
                          AND a.aspect = t.aspect AND a.sentiment = t.polarity
            WHERE te.theme_id = ? AND a.severity = 3
        """, [t["theme_id"]]).fetchone()[0]
        return {"metrics": {
            "n_reviews": t["n_reviews"],
            "volume_norm": t["n_reviews"] / max_n,
            "share_of_aspect": round(t["n_reviews"] / aspect_reviews, 3) if aspect_reviews else 0.0,
            "neg_share": round(neg_mentions / all_mentions, 3) if all_mentions else 0.0,
            "recent_count": int(recent or 0),
            "prior_count": int(prior or 0),
            "trend": round(((recent or 0) - (prior or 0)) / max(prior or 0, 1), 3),
            "safety_flag": bool(safety) and t["polarity"] == "negative",
        }}

    def score(state: InsightState) -> dict:
        m, t = state["metrics"], state["theme"]
        s = (W_VOLUME * m["volume_norm"] + W_NEG * m["neg_share"]
             + W_SAFETY * float(m["safety_flag"]) + W_TREND * min(max(m["trend"], 0.0), 1.0))
        if t["polarity"] != "negative":
            priority = "P3"
        elif m["safety_flag"]:
            priority = "P1"
        elif s >= P2_THRESHOLD:
            priority = "P2"
        else:
            priority = "P3"
        return {"severity_score": round(s, 3), "priority": priority}

    def route(state: InsightState) -> dict:
        t = state["theme"]
        if t["polarity"] == "positive":
            return {"team": RULES["positive_team"], "rationale": "Positive theme: candidate proof point / claim."}
        text = (t["label"] + " " + t["summary"]).lower()
        for o in RULES["keyword_overrides"]:
            if any(k in text for k in o["keywords"]):
                return {"team": o["team"], "rationale": f"Keyword rule matched ({', '.join(o['keywords'][:3])}…)."}
        rule = RULES["aspects"].get(t["aspect"], {"team": "Consumer insights"})
        if not rule.get("ambiguous"):
            return {"team": rule["team"], "rationale": f"Aspect rule: {t['aspect']}."}
        candidates = rule["candidates"]
        if not use_llm():
            return {"team": candidates[0], "rationale": "Ambiguous aspect; baseline picks the first candidate."}
        try:
            d = get_llm().structured(RoutingDecision, (
                f"Theme for {t['product_id']}: {t['label']}\nSummary: {t['summary']}\n"
                f"Pick exactly one owning team from {candidates}. If consumers say the product doesn't work, "
                f"that's R&D; if they say it was oversold or results differ from the claim, that's Marketing & claims."
            ), tier="fast", max_tokens=200)
            team = d.team if d.team in candidates else candidates[0]
            return {"team": team, "rationale": d.rationale}
        except Exception as e:
            return {"team": candidates[0], "rationale": f"LLM routing failed ({e}); default."}

    def save_metrics(state: InsightState) -> dict:
        m, t = state["metrics"], state["theme"]
        con.execute("INSERT OR REPLACE INTO theme_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            t["theme_id"], m["n_reviews"], m["neg_share"], m["share_of_aspect"], m["recent_count"],
            m["prior_count"], m["trend"], m["safety_flag"], state["severity_score"], state["priority"],
            state["team"], state["rationale"],
        ])
        return {}

    def human_review(state: InsightState) -> dict:
        t, prior, track = state["theme"], state.get("prior"), state.get("track") or {}
        if prior and track.get("status") != "growing":
            note = prior["reviewer_note"] or ""
            if not note.startswith("Carried forward"):  # keep the reviewer's own words, once
                note = (f"Carried forward: {prior['status']} on {str(prior['updated_at'])[:10]}, not re-asked "
                        f"while the issue isn't growing." + (f" Original note: {note}" if note else ""))
            return {"decision": {"action": "approve" if prior["status"] == "approved" else "reject",
                                 "note": note}, "carried_forward": True}
        payload = {
            "theme_id": t["theme_id"], "product_id": t["product_id"], "label": t["label"],
            "summary": t["summary"], "team": state["team"], "priority": state["priority"],
            "severity_score": state["severity_score"], "metrics": state["metrics"],
        }
        if prior:  # asked again because the issue grew since the last decision
            payload["previous_decision"] = {
                "status": prior["status"], "note": prior["reviewer_note"], "on": str(prior["updated_at"])[:10],
                "reviews_then": track.get("prev_n_reviews"), "reviews_now": track.get("n_reviews"),
            }
        decision = interrupt(payload)
        return {"decision": decision}

    def create_ticket(state: InsightState) -> dict:
        t, d, prior = state["theme"], state.get("decision"), state.get("prior")
        if d is None:
            status, note = (prior["status"], prior["reviewer_note"]) if prior else ("auto_created", None)
        else:
            status = {"approve": "approved", "reject": "rejected"}.get(d.get("action"), "pending_review")
            note = d.get("note")
        if prior:  # same issue as an existing ticket: update it in place, keep its id
            if status == "pending_review":  # a deferred re-review must not erase the earlier decision
                status, note = prior["status"], prior["reviewer_note"]
            con.execute("""
                UPDATE tickets SET theme_id = ?, theme_label = ?, team = ?, priority = ?, severity_score = ?,
                       status = ?, reviewer_note = ?, updated_at = ?
                WHERE ticket_id = ?
            """, [t["theme_id"], t["label"], state["team"], state["priority"], state["severity_score"],
                  status, note, now(), prior["ticket_id"]])
            return {"ticket_id": prior["ticket_id"], "status": status}
        tid = new_id("tk_")
        con.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            tid, t["theme_id"], t["product_id"], t["label"], state["team"], state["priority"],
            state["severity_score"], status, note, now(), now(),
        ])
        return {"ticket_id": tid, "status": status}

    def after_save(state: InsightState) -> str:
        return {"P1": "human_review", "P2": "create_ticket"}.get(state["priority"], END)

    g = StateGraph(InsightState)
    for name, fn in [("compute_metrics", compute_metrics), ("score", score), ("route", route),
                     ("save_metrics", save_metrics), ("human_review", human_review),
                     ("create_ticket", create_ticket)]:
        g.add_node(name, fn)
    g.add_edge(START, "compute_metrics")
    g.add_edge("compute_metrics", "score")
    g.add_edge("score", "route")
    g.add_edge("route", "save_metrics")
    g.add_conditional_edges("save_metrics", after_save, ["human_review", "create_ticket", END])
    g.add_edge("human_review", "create_ticket")
    g.add_edge("create_ticket", END)
    return g.compile(checkpointer=InMemorySaver())


Reviewer = Callable[[dict], dict]


def run(con, reviewer: Reviewer | None = None, auto_approve: bool = False) -> dict:
    """reviewer(payload) -> {"action": "approve"|"reject"|"defer", "note": str}.

    No reviewer and no auto_approve => P1 tickets are filed as pending_review (resolve later with `voc review`).
    """
    graph = build_graph(con)
    with run_step(con, "insights") as (_, stats):
        con.execute("DELETE FROM tickets WHERE status IN ('auto_created', 'pending_review')")
        # Human decisions survive re-runs: index them by the track (issue) they belong to.
        priors = {r["track_id"]: r for r in rows(con, """
            SELECT s.track_id, tk.ticket_id, tk.status, tk.reviewer_note, tk.updated_at
            FROM tickets tk JOIN theme_track_runs s ON s.theme_id = tk.theme_id
            WHERE tk.status IN ('approved', 'rejected')
            QUALIFY row_number() OVER (PARTITION BY s.track_id ORDER BY tk.updated_at DESC) = 1
        """)}
        themes = rows(con, "SELECT * FROM themes ORDER BY product_id, n_reviews DESC")
        counts: dict[str, Any] = {"themes": len(themes), "P1": 0, "P2": 0, "P3": 0, "tickets": {},
                                  "decisions_carried_forward": 0, "re_reviewed_because_growing": 0}
        for t in themes:
            t = {k: (str(v) if k == "created_at" else v) for k, v in t.items()}
            cfg = {"configurable": {"thread_id": t["theme_id"]}}
            track = track_of(con, t["theme_id"]) or {}
            track = {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in track.items()}
            prior = priors.get(track.get("track_id"))
            if prior:
                prior = {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in prior.items()}
            out = graph.invoke({"theme": t, "track": track, "prior": prior}, cfg)
            if "__interrupt__" in out:
                payload = out["__interrupt__"][0].value
                if reviewer:
                    decision = reviewer(payload)
                elif auto_approve:
                    decision = {"action": "approve", "note": "auto-approved (demo mode)"}
                else:
                    decision = {"action": "defer", "note": None}
                out = graph.invoke(Command(resume=decision), cfg)
                if prior:
                    counts["re_reviewed_because_growing"] += 1
            if out.get("carried_forward"):
                counts["decisions_carried_forward"] += 1
            if prior and not out.get("ticket_id"):  # dropped below P2: keep the decision linked to the issue
                con.execute("UPDATE tickets SET theme_id = ?, theme_label = ? WHERE ticket_id = ?",
                            [t["theme_id"], t["label"], prior["ticket_id"]])
            counts[out["priority"]] += 1
            if out.get("status"):
                counts["tickets"][out["status"]] = counts["tickets"].get(out["status"], 0) + 1
        stats.update(counts)
    return stats


def resolve_pending(con, reviewer: Reviewer) -> int:
    pending = rows(con, """
        SELECT tk.*, t.summary FROM tickets tk JOIN themes t USING (theme_id)
        WHERE tk.status = 'pending_review' ORDER BY tk.severity_score DESC
    """)
    for tk in pending:
        d = reviewer(tk)
        status = {"approve": "approved", "reject": "rejected"}.get(d.get("action"))
        if status:
            con.execute("UPDATE tickets SET status = ?, reviewer_note = ?, updated_at = ? WHERE ticket_id = ?",
                        [status, d.get("note"), now(), tk["ticket_id"]])
    return len(pending)
