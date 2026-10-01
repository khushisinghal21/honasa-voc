"""Minimal HTTP API over the same service layer (handy for a dashboard or a quick curl demo)."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from voc.serve import service

app = FastAPI(title="Honasa Voice of Consumer POC", version="0.1.0")


class SQLRequest(BaseModel):
    sql: str


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/products")
def products():
    return service.list_products()


@app.get("/themes")
def themes(product_id: str | None = None, aspect: str | None = None, polarity: str = "negative", limit: int = 10):
    return service.top_themes(product_id, aspect, polarity, limit)


@app.get("/themes/{theme_id}")
def theme(theme_id: str):
    out = service.theme_detail(theme_id)
    if "error" in out:
        raise HTTPException(404, out["error"])
    return out


@app.get("/search")
def search(q: str, product_id: str | None = None, k: int = 10):
    return service.search_reviews(q, product_id, k)


@app.post("/sql")
def sql(req: SQLRequest):
    out = service.run_metrics_sql(req.sql)
    if "error" in out:
        raise HTTPException(400, out["error"])
    return out


@app.get("/digest")
def digest(period: str | None = None):
    return service.get_digest(period)
