"""Guardrails for the NL→SQL tool. Defence in depth:
  1. parse with sqlglot: exactly one statement, and it must be a SELECT (CTEs and UNIONs allowed)
  2. every table referenced must be on the allowlist (or a CTE defined in the query)
  3. no file/network/system functions (read_csv, read_parquet, glob, getenv, ...)
  4. a row cap is forced by wrapping the query
  5. it then runs on a read-only DuckDB connection anyway
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from voc.db import QUERYABLE_TABLES

MAX_ROWS = 200
BLOCKED_FUNCS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto", "read_text", "read_blob",
    "read_ndjson", "glob", "getenv", "current_setting", "duckdb_settings", "duckdb_secrets",
    "duckdb_extensions", "parquet_scan", "csv_scan", "sniff_csv", "query", "query_table",
}


class UnsafeSQL(ValueError):
    pass


def validate(sql: str, max_rows: int = MAX_ROWS) -> str:
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        raise UnsafeSQL("empty query")
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQL(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQL("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise UnsafeSQL(f"only SELECT queries are allowed (got {type(tree).__name__})")

    for node in tree.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter,
                             exp.Command, exp.Copy, exp.Set, exp.Pragma)):
            raise UnsafeSQL(f"{type(node).__name__} is not allowed")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in QUERYABLE_TABLES} | cte_names
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Func) or table.args.get("this") is None:
            raise UnsafeSQL("table functions are not allowed")
        if table.args.get("db") or table.args.get("catalog"):
            raise UnsafeSQL("schema/catalog-qualified tables are not allowed")
        if table.name.lower() not in allowed:
            raise UnsafeSQL(f"table '{table.name}' is not queryable. Allowed: {', '.join(QUERYABLE_TABLES)}")

    for func in tree.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name in BLOCKED_FUNCS or name.startswith("read_"):
            raise UnsafeSQL(f"function '{name}' is not allowed")

    inner = tree.sql(dialect="duckdb")
    return f"SELECT * FROM ({inner}) AS q LIMIT {int(max_rows)}"
