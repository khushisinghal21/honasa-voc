# Honasa Voice of Consumer agent (POC)

Consumers talk about Mamaearth, The Derma Co and Aqualogica products every day in YouTube comments, Reddit threads and reviews. That feedback is scattered, unstructured, often in Hinglish, and too large to read. Signals like *"the pump jams halfway"*, *"broke me out"* or *"I think I got a fake"* reach the right team late, or never.

This POC reads that feedback and turns it into **themes backed by verbatim quotes**. Each theme is **routed to the team that owns it**, with a human approval step for safety issues. The whole dataset is **queryable from Claude through MCP**, with cited answers.

It's a working slice of the *Voice of Consumer OS* and *Business Data MCP* ideas from Honasa's AI & Analytics roadmap.

```
YouTube · Reddit · datasets
        │
        ▼
 clean ─── PII scrub, en/hi/Hinglish tagging, dedupe
        │
        ▼
 extract ─ Claude, strict schema → aspect · sentiment · severity · evidence span
        │   every span verified verbatim in code; paraphrases are dropped
        ▼
 themes ── multilingual embeddings of spans → HDBSCAN → Claude labels each cluster
        │
        ▼
 insights ─ LangGraph: metrics → score → route → P1 pauses for human approval → ticket
        │
        ▼
 digest + MCP server (9 tools) + HTTP API          all backed by one DuckDB file
```

## What it does

| Stage | What happens | Why it matters |
|---|---|---|
| **Collect** | YouTube Data API v3 and Reddit API (PRAW), plus CSV/JSONL import | Official APIs only. Author IDs are stored as salted hashes, never names |
| **Clean** | Strip emails/phones/@mentions, tag `en`/`hi`/`hinglish`, exact + near-duplicate removal | Hinglish is a big share of Indian beauty chatter |
| **Extract** | Claude fills a Pydantic schema: relevance, 8 aspects, sentiment, severity 1–3, evidence span | Structured output that SQL can count |
| **Verify** | Each `evidence_span` must be a verbatim substring of the comment (case/whitespace tolerant only) | **No theme can cite words a consumer didn't write** |
| **Themes** | Embed spans (not whole reviews), cluster per product × aspect × polarity, Claude names each cluster | One review can feed a fragrance theme *and* a packaging theme |
| **Insights** | LangGraph scores volume, negativity, safety and 4-week trend, routes by rules (Claude breaks ties), files tickets | P1 safety/trust issues stop at `interrupt()` until a person approves |
| **Serve** | MCP tools + FastAPI over the same service layer | Ask Claude "what's going wrong with Rosemary shampoo?" and get quotes with links |

Aspects: `efficacy`, `skin_reaction`, `fragrance`, `texture`, `packaging`, `price_value`, `authenticity`, `delivery`.

## Quick start (no API keys)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
voc demo          # synthetic data → full pipeline → digest + evals in reports/demo/
pytest -q         # 30 tests
```

`voc demo` runs on **synthetic template data** with the rule-based baseline and TF-IDF embeddings, so it needs no keys or downloads. It proves the plumbing works. It says nothing about real products, and every output from it is labelled as synthetic.

## Real run

**1. Keys.** Copy `.env.example` to `.env` and fill in:
- `ANTHROPIC_API_KEY`
- `YOUTUBE_API_KEY`: Google Cloud Console → enable *YouTube Data API v3* → create an API key
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET`: reddit.com/prefs/apps → create a *script* app
- `VOC_AUTHOR_SALT`: any random string

```bash
pip install -e ".[ml,reddit,dev]"   # multilingual embeddings + PRAW
```

**2. Pick sources.** Edit `data/products.yaml`. Pin 2–3 good review videos per product in `video_ids`. Each YouTube search costs 100 of your 10,000 daily quota units; fetching comments costs about 1 unit per 100.

**3. Collect and clean.**
```bash
voc init
voc collect youtube
voc collect reddit
voc clean
voc stats            # aim for ≥200 clean comments per product; swap products that fall short
```

**4. Extract, starting small.**
```bash
voc extract --limit 50   # check the cost line and spot-check the aspects table
voc extract              # the rest
```

**5. Themes, insights, digest.**
```bash
voc themes
voc insights             # prompts you to approve/reject each P1 theme
voc digest               # reports/digest_<week>.md
```
Or run it all at once: `voc run`. Use `--no-input` to leave P1s as `pending_review`, then resolve them later with `voc review`.

## Use it from Claude Desktop

Add this to `claude_desktop_config.json` (Settings → Developer → Edit config). Use absolute paths, then restart Claude Desktop:

```json
{
  "mcpServers": {
    "honasa-voc": {
      "command": "/ABSOLUTE/PATH/honasa-voc/.venv/bin/voc",
      "args": ["serve", "mcp"],
      "env": { "VOC_DB": "/ABSOLUTE/PATH/honasa-voc/data/voc.duckdb" }
    }
  }
}
```

Questions to try:
- *What are the top problems with Mamaearth Rosemary shampoo? Quote consumers.*
- *Compare Aqualogica and Minimalist sunscreen on texture and skin reaction.*
- *Which packaging complaints grew in the last 4 weeks?*
- *What share of Tea Tree face wash reviews mention dryness?* (Claude writes SQL against `get_schema`)

| MCP tool | Purpose |
|---|---|
| `list_products` | Products, IDs, review counts |
| `top_themes` | Ranked themes by product / aspect / polarity |
| `theme_detail` | Metrics, routing, tickets, and verbatim quotes with source links |
| `compare_products` | Aspect sentiment side by side, per 100 relevant reviews |
| `search_reviews` | Hybrid BM25 + dense search with reciprocal rank fusion |
| `get_schema` / `run_metrics_sql` | Guarded, read-only NL→SQL |
| `list_tickets`, `get_digest` | Actions raised and the weekly digest |

HTTP API: `voc serve api` → `/products`, `/themes`, `/themes/{id}`, `/search`, `/sql`, `/digest`.

## Evaluation

```bash
voc label --n 150                                # hand-label a stratified sample (~40 min)
voc eval extraction --backend baseline           # floor
voc eval extraction --backend anthropic          # Claude
voc eval themes                                  # LLM-as-judge faithfulness + cluster cohesion
voc eval sql                                     # guard red-team + NL→SQL execution accuracy
```

Reports land in `reports/` as Markdown + JSON. Fill in this table from your run:

| Metric | Baseline | Claude |
|---|---|---|
| Relevance accuracy | | |
| Aspect detection F1 (micro / macro) | | |
| Aspect + sentiment F1 | | |
| Evidence spans verified verbatim | 100% by construction | |
| Theme faithfulness (LLM judge) | n/a | |
| NL→SQL execution accuracy | n/a | |
| SQL guard: attacks blocked | 10/10 | 10/10 |
| Cost per 1k comments | $0 | |

Use the baseline-vs-Claude comparison to show where the LLM earns its cost. The baseline can't tell that *"paisa waste ho gaya"* in a hair-fall complaint is about efficacy, not price.

## Design decisions

- **Verbatim evidence over trust.** A model asked for quotes will sometimes paraphrase or translate them. Checking spans in code turns that into a measurable metric instead of a silent risk.
- **Embed spans, not reviews.** Mixed reviews are the norm, and whole-review embeddings blur them into one muddy theme.
- **HDBSCAN.** It doesn't need the number of themes up front, and it leaves one-off comments out as noise instead of forcing them into a theme.
- **Rules first, LLM second for routing.** Ownership should be predictable and auditable. Claude only decides genuinely ambiguous cases (efficacy: R&D vs claims) and must give a rationale.
- **Human-in-the-loop only where it matters.** Safety and authenticity themes pause for approval; routine themes auto-file. Every decision is stored in `tickets`.
- **Defence in depth on SQL.** sqlglot allowlist (SELECT only, approved tables, no file/env functions), forced row cap, *and* a read-only connection.
- **One DuckDB file.** Every stage is inspectable with SQL, and the same schema maps directly to BigQuery.

## Limitations

- Public comments over-represent strong opinions. These are **signals to investigate**, not market-share numbers.
- YouTube comments on multi-product videos are noisy. The relevance check filters most of it, but not all.
- Marketplace and quick-commerce reviews (the richest source) aren't included, because scraping them breaks their terms. Inside Honasa this would plug into first-party review feeds.
- Themes are rebuilt from scratch each run. Tracking themes week over week would need cluster matching across runs.
- The HTTP MCP server has no auth. Deploy it private (below) or add an auth provider before exposing it.

## Deploy (Cloud Run)

Run the pipeline locally first, since the image ships `data/voc.duckdb` read-only.

```bash
gcloud run deploy honasa-voc --source . --region asia-south1 --no-allow-unauthenticated
```
The MCP endpoint is then `https://<service-url>/mcp`.

## Layout

```
voc/
  collectors/   youtube.py · reddit.py · file_import.py
  pipeline/     clean.py · extract.py · themes.py · digest.py
  graph/        insight_graph.py · routing_rules.yaml
  retrieval/    hybrid.py
  serve/        service.py · mcp_server.py · api.py · sql_guard.py
  evaluation.py · labeling.py · llm.py · embeddings.py · schemas.py · db.py · cli.py
evals/          sql_questions.yaml · gold_labels.jsonl (yours)
scripts/        make_synthetic.py
tests/          test_core.py · test_llm_path.py
```
