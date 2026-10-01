"""Embeddings with a graceful fallback.

- "st": sentence-transformers (default model is multilingual, so Hinglish clusters with English).
- "tfidf": TF-IDF (word + char n-grams) reduced with SVD. No downloads; used in offline demo and tests.
"""
from __future__ import annotations

import logging

import numpy as np

from voc.config import settings

log = logging.getLogger(__name__)
_st_model = None


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_st():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(settings.embed_model)
    return _st_model


def backend() -> str:
    if settings.embed_backend in ("st", "tfidf"):
        return settings.embed_backend
    try:
        _load_st()
        return "st"
    except Exception as e:  # missing package, no network, etc.
        log.warning("sentence-transformers unavailable (%s); falling back to TF-IDF embeddings", e)
        return "tfidf"


def embed(texts: list[str], kind: str = "passage") -> np.ndarray:
    """Embed a self-contained set of texts. With TF-IDF the space is fit on `texts` itself."""
    if not texts:
        return np.zeros((0, 1))
    if backend() == "st":
        model = _load_st()
        prefix = f"{kind}: " if "e5" in settings.embed_model.lower() else ""
        vecs = model.encode([prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs)
    return _normalize(TfidfSpace(texts).transform(texts))


class TfidfSpace:
    """A fitted TF-IDF+SVD space, so queries can be embedded into the same space as a corpus."""

    def __init__(self, corpus: list[str], dims: int = 128):
        from scipy.sparse import hstack
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._hstack = hstack
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        m = hstack([self.word.fit_transform(corpus), self.char.fit_transform(corpus)]).tocsr()
        k = max(2, min(dims, m.shape[0] - 1, m.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0) if m.shape[0] > 2 else None
        if self.svd is not None:
            import warnings

            with warnings.catch_warnings():  # tiny corpora trigger harmless variance-ratio warnings
                warnings.simplefilter("ignore", RuntimeWarning)
                self.svd.fit(m)

    def transform(self, texts: list[str]) -> np.ndarray:
        m = self._hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        out = self.svd.transform(m) if self.svd is not None else m.toarray()
        return _normalize(np.asarray(out))


class Embedder:
    """Corpus-aware embedder for retrieval: queries land in the same space as the documents."""

    def __init__(self, corpus: list[str]):
        self.kind = backend()
        if self.kind == "tfidf":
            self.space = TfidfSpace(corpus)
            self.doc_vecs = self.space.transform(corpus)
        else:
            self.doc_vecs = embed(corpus, kind="passage")

    def query(self, q: str) -> np.ndarray:
        if self.kind == "tfidf":
            return self.space.transform([q])[0]
        return embed([q], kind="query")[0]
