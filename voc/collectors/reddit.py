"""Reddit collector via the official API (PRAW). Create a "script" app at reddit.com/prefs/apps.

A comment is kept if it mentions the product itself, or if the thread title/body is about the product
(so short replies like "same, it broke me out" are kept in context). The LLM relevance check filters the rest.
"""
from __future__ import annotations

import logging

from voc.collectors.common import author_hash, insert_raw, mentions_product, parse_ts
from voc.config import settings

log = logging.getLogger(__name__)


def _client():
    try:
        import praw
    except ImportError as e:
        raise RuntimeError("pip install praw") from e
    if not (settings.reddit_client_id and settings.reddit_client_secret):
        raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set in .env")
    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
        check_for_async=False,
    )


def _row(kind: str, obj_id: str, permalink: str, product_id: str, text: str, author, created_utc, score, title):
    return {
        "review_id": f"rd:{kind}:{obj_id}",
        "source": "reddit",
        "source_url": f"https://www.reddit.com{permalink}",
        "product_id": product_id,
        "text": text,
        "author_hash": author_hash(str(author) if author else None),
        "created_at": parse_ts(created_utc),
        "likes": score,
        "parent_context": title,
    }


def collect(con, products: list[dict], posts_per_query: int = 25, time_filter: str = "year") -> dict:
    reddit, stats = _client(), {}
    for p in products:
        cfg, rows, seen = p.get("reddit", {}), [], set()
        for sub in cfg.get("subreddits", []):
            for q in cfg.get("queries", []):
                try:
                    results = reddit.subreddit(sub).search(q, sort="relevance", time_filter=time_filter,
                                                           limit=posts_per_query)
                    for post in results:
                        if post.id in seen:
                            continue
                        seen.add(post.id)
                        thread_text = f"{post.title}\n{post.selftext or ''}"
                        thread_on_topic = mentions_product(thread_text, p)
                        if thread_on_topic and post.selftext:
                            rows.append(_row("t3", post.id, post.permalink, p["id"], thread_text, post.author,
                                             post.created_utc, post.score, post.title))
                        post.comments.replace_more(limit=0)
                        for c in post.comments.list():
                            body = c.body or ""
                            if body in ("[deleted]", "[removed]"):
                                continue
                            if thread_on_topic or mentions_product(body, p):
                                rows.append(_row("t1", c.id, c.permalink, p["id"], body, c.author,
                                                 c.created_utc, c.score, post.title))
                except Exception as e:  # private sub, rate limit, etc.
                    log.warning("reddit r/%s %r failed: %s", sub, q, e)
        added = insert_raw(con, rows)
        stats[p["id"]] = {"threads": len(seen), "items_added": added}
        log.info("reddit %s: %s", p["id"], stats[p["id"]])
    return stats
