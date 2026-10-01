"""Hybrid review search: BM25 (exact words, product names, Hinglish spellings) + dense (meaning),
merged with reciprocal rank fusion. Same approach as the codebase copilot's retriever."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from voc.embeddings import Embedder

TOKEN_RE = re.compile(r"[a-z0-9\u0900-\u097f]+")
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class HybridSearcher:
    def __init__(self, docs: list[dict]):
        """docs: dicts with at least review_id, product_id, text."""
        self.docs = docs
        texts = [d["text"] for d in docs] or [""]
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in texts])
        self.embedder = Embedder(texts)

    def search(self, query: str, k: int = 10, product_id: str | None = None, pool: int = 50) -> list[dict]:
        if not self.docs:
            return []
        allowed = np.array([product_id is None or d["product_id"] == product_id for d in self.docs])
        bm = np.where(allowed, self.bm25.get_scores(tokenize(query) or ["_"]), -np.inf)
        dense = np.where(allowed, self.embedder.doc_vecs @ self.embedder.query(query), -np.inf)
        fused: dict[int, float] = {}
        for scores in (bm, dense):
            ranking = np.argsort(-scores)[:pool]
            for rank, idx in enumerate(ranking):
                if allowed[idx] and scores[idx] > 0:
                    fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [{**self.docs[i], "score": round(s, 5)} for i, s in top]
