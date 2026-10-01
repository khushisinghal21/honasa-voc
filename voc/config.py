"""Central configuration. Everything is driven by environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class Settings:
    db_path: Path = field(default_factory=lambda: ROOT / _env("VOC_DB", "data/voc.duckdb"))
    products_file: Path = field(default_factory=lambda: ROOT / _env("VOC_PRODUCTS", "data/products.yaml"))
    reports_dir: Path = field(default_factory=lambda: ROOT / _env("VOC_REPORTS", "reports"))

    # "anthropic" = real LLM calls; "baseline" = rule-based extractor (no API key needed).
    llm_backend: str = field(default_factory=lambda: _env("VOC_LLM_BACKEND", "anthropic"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    fast_model: str = field(default_factory=lambda: _env("VOC_FAST_MODEL", "claude-haiku-4-5-20251001"))
    smart_model: str = field(default_factory=lambda: _env("VOC_SMART_MODEL", "claude-sonnet-5-5"))
    # USD per million tokens, used only for the cost report. Check current pricing and override.
    price_fast_in: float = field(default_factory=lambda: float(_env("VOC_PRICE_FAST_IN", "1.0")))
    price_fast_out: float = field(default_factory=lambda: float(_env("VOC_PRICE_FAST_OUT", "5.0")))
    price_smart_in: float = field(default_factory=lambda: float(_env("VOC_PRICE_SMART_IN", "3.0")))
    price_smart_out: float = field(default_factory=lambda: float(_env("VOC_PRICE_SMART_OUT", "15.0")))

    # "auto" tries sentence-transformers, falls back to TF-IDF+SVD. "st" or "tfidf" to force.
    embed_backend: str = field(default_factory=lambda: _env("VOC_EMBED_BACKEND", "auto"))
    embed_model: str = field(default_factory=lambda: _env("VOC_EMBED_MODEL", "intfloat/multilingual-e5-small"))

    extract_batch_size: int = field(default_factory=lambda: int(_env("VOC_BATCH_SIZE", "20")))
    llm_workers: int = field(default_factory=lambda: int(_env("VOC_LLM_WORKERS", "4")))

    youtube_api_key: str = field(default_factory=lambda: _env("YOUTUBE_API_KEY"))
    reddit_client_id: str = field(default_factory=lambda: _env("REDDIT_CLIENT_ID"))
    reddit_client_secret: str = field(default_factory=lambda: _env("REDDIT_CLIENT_SECRET"))
    reddit_user_agent: str = field(default_factory=lambda: _env("REDDIT_USER_AGENT", "honasa-voc-poc/0.1"))
    author_salt: str = field(default_factory=lambda: _env("VOC_AUTHOR_SALT", "change-me"))


settings = Settings()


def load_products(path: Path | None = None) -> list[dict]:
    data = yaml.safe_load((path or settings.products_file).read_text())
    return data["products"]


def product_map(path: Path | None = None) -> dict[str, dict]:
    return {p["id"]: p for p in load_products(path)}
