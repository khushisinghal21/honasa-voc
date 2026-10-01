"""Stage 5: a weekly Markdown digest. Every theme is backed by quoted, linked consumer evidence."""
from __future__ import annotations

from datetime import date

from voc.config import settings
from voc.db import now, rows, run_step

QUOTES_PER_THEME = 3
NEG_THEMES_PER_PRODUCT = 4
POS_THEMES_PER_PRODUCT = 2


def _quotes(con, theme_id: str, n: int = QUOTES_PER_THEME) -> list[dict]:
    """Top-ranked quotes, skipping exact repeats so each line adds information."""
    out, seen = [], set()
    for q in rows(con, """
        SELECT te.evidence_span, r.source, r.source_url, r.review_id
        FROM theme_evidence te JOIN reviews r USING (review_id)
        WHERE te.theme_id = ? ORDER BY te.rank LIMIT 50
    """, [theme_id]):
        key = q["evidence_span"].lower().strip(" .!")
        if key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) == n:
            break
    return out


def _fmt_quote(q: dict) -> str:
    cite = f"[{q['source']}]({q['source_url']})" if q.get("source_url") else f"`{q['review_id']}`"
    return f'  - "{q["evidence_span"]}" ({cite})'


def build(con, period: str | None = None) -> str:
    period = period or f"{date.today().isocalendar().year}-W{date.today().isocalendar().week:02d}"
    sources = rows(con, "SELECT source, count(*) AS n FROM reviews GROUP BY 1 ORDER BY 2 DESC")
    window = con.execute("SELECT min(created_at), max(created_at) FROM reviews").fetchone()
    relevant = con.execute("SELECT count(*) FROM extractions WHERE is_relevant").fetchone()[0]
    backend = con.execute("SELECT any_value(backend) FROM extractions").fetchone()[0]

    lines = [f"# Voice of Consumer digest · {period}", ""]
    if any(s["source"] == "synthetic" for s in sources):
        lines += ["> **Synthetic demo data.** These reviews were generated to test the pipeline. "
                  "Nothing here reflects real consumer opinion.", ""]
    source_mix = ", ".join(f"{s['source']} {s['n']}" for s in sources)
    lines += [
        f"Reviews analysed: **{sum(s['n'] for s in sources)}** "
        f"({source_mix}), of which **{relevant}** were relevant. "
        f"Window: {window[0]:%d %b %Y} – {window[1]:%d %b %Y}. Extraction backend: `{backend}`."
        if window[0] else "No reviews yet.",
        "",
    ]

    tickets = rows(con, """
        SELECT tk.priority, tk.team, tk.theme_label, tk.status, p.name
        FROM tickets tk JOIN products p USING (product_id)
        ORDER BY tk.priority, tk.severity_score DESC
    """)
    if tickets:
        lines += ["## Actions raised", "", "| Priority | Product | Theme | Owner | Status |", "|---|---|---|---|---|"]
        lines += [f"| {t['priority']} | {t['name']} | {t['theme_label']} | {t['team']} | {t['status']} |" for t in tickets]
        lines.append("")

    for p in rows(con, "SELECT * FROM products ORDER BY is_competitor, name"):
        neg = rows(con, """
            SELECT t.*, m.priority, m.severity_score, m.team, m.trend, m.recent_count, m.prior_count
            FROM themes t LEFT JOIN theme_metrics m USING (theme_id)
            WHERE t.product_id = ? AND t.polarity = 'negative'
            ORDER BY m.severity_score DESC NULLS LAST, t.n_reviews DESC LIMIT ?
        """, [p["product_id"], NEG_THEMES_PER_PRODUCT])
        pos = rows(con, """
            SELECT * FROM themes WHERE product_id = ? AND polarity = 'positive'
            ORDER BY n_reviews DESC LIMIT ?
        """, [p["product_id"], POS_THEMES_PER_PRODUCT])
        if not (neg or pos):
            continue
        tag = " (competitor baseline)" if p["is_competitor"] else ""
        lines += [f"## {p['name']}{tag}", ""]
        if neg:
            lines.append("**What's going wrong**")
            for t in neg:
                trend = ""
                if t.get("recent_count") is not None:
                    trend = f", last 4 wks {t['recent_count']} vs prior {t['prior_count']}"
                lines.append(f"- **{t['label']}** · {t['n_reviews']} reviews · {t.get('priority') or '–'}"
                             f" → {t.get('team') or '–'}{trend}  \n  {t['summary']}")
                lines += [_fmt_quote(q) for q in _quotes(con, t["theme_id"])]
            lines.append("")
        if pos:
            lines.append("**What consumers love**")
            for t in pos:
                lines.append(f"- **{t['label']}** · {t['n_reviews']} reviews  \n  {t['summary']}")
                lines += [_fmt_quote(q) for q in _quotes(con, t["theme_id"], 2)]
            lines.append("")
    return "\n".join(lines)


def run(con, period: str | None = None) -> dict:
    with run_step(con, "digest") as (_, stats):
        md = build(con, period)
        period = md.splitlines()[0].split("·")[-1].strip()
        con.execute("INSERT OR REPLACE INTO digests VALUES (?, ?, ?)", [period, md, now()])
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        path = settings.reports_dir / f"digest_{period}.md"
        path.write_text(md, encoding="utf-8")
        stats.update({"period": period, "path": str(path), "chars": len(md)})
    return stats
