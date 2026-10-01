"""MCP server: lets Claude (Desktop, Code, or any MCP client) query the Voice of Consumer data
in plain English, with cited answers."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from voc.serve import service

INSTRUCTIONS = """Voice of Consumer data for Honasa products, built from public consumer comments.
Workflow: call list_products first to get product_ids. Use top_themes for "what are people saying",
theme_detail to get verbatim quotes with source links, compare_products for side-by-side questions,
search_reviews for anything specific. For custom numbers call get_schema, then run_metrics_sql
(read-only SELECT only). When you state a finding, cite at least one quote and its source link.
Counts come from public comments, which skew toward people with strong opinions; say so when relevant."""

mcp = MCPServer("honasa-voc", instructions=INSTRUCTIONS)
RO = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


@mcp.tool(annotations=RO)
def list_products() -> list[dict]:
    """List tracked products with product_id, brand, category, and review counts."""
    return service.list_products()


@mcp.tool(annotations=RO)
def top_themes(product_id: str | None = None, aspect: str | None = None,
               polarity: str = "negative", limit: int = 10) -> list[dict]:
    """Top consumer themes, ranked by severity score. polarity: 'negative' (problems) or 'positive' (what people love).
    aspect (optional): efficacy, skin_reaction, fragrance, texture, packaging, price_value, authenticity, delivery."""
    return service.top_themes(product_id, aspect, polarity, limit)


@mcp.tool(annotations=RO)
def theme_detail(theme_id: str, max_quotes: int = 15) -> dict:
    """One theme's metrics, routing decision, any tickets, and the verbatim consumer quotes behind it (with source links)."""
    return service.theme_detail(theme_id, max_quotes)


@mcp.tool(annotations=RO)
def compare_products(product_a: str, product_b: str) -> dict:
    """Side-by-side aspect sentiment for two products, normalised per 100 relevant reviews."""
    return service.compare_products(product_a, product_b)


@mcp.tool(annotations=RO)
def search_reviews(query: str, product_id: str | None = None, k: int = 10) -> list[dict]:
    """Hybrid keyword + semantic search over relevant consumer comments. Works with Hinglish queries."""
    return service.search_reviews(query, product_id, k)


@mcp.tool(annotations=RO)
def get_schema() -> dict:
    """Tables and columns available to run_metrics_sql, plus notes on allowed values."""
    return service.get_schema()


@mcp.tool(annotations=RO)
def run_metrics_sql(sql: str) -> dict:
    """Run ONE read-only DuckDB SELECT over the allowlisted tables (call get_schema first). Results are capped at 200 rows."""
    return service.run_metrics_sql(sql)


@mcp.tool(annotations=RO)
def list_tickets(status: str | None = None) -> list[dict]:
    """Action tickets raised from themes. status: approved, rejected, auto_created, pending_review."""
    return service.list_tickets(status)


@mcp.tool(annotations=RO)
def get_digest(period: str | None = None) -> dict:
    """The latest (or a given period's, e.g. '2026-W40') weekly digest in Markdown."""
    return service.get_digest(period)


def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    if transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
