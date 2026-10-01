"""YouTube Data API v3 collector (official API, API key only).

Quota: search.list costs 100 units, commentThreads.list costs 1 unit per page of up to 100 comments.
The default daily quota is 10,000 units, so prefer pinning video_ids in products.yaml over searching.
"""
from __future__ import annotations

import logging

import requests

from voc.collectors.common import author_hash, insert_raw, parse_ts
from voc.config import settings

log = logging.getLogger(__name__)
API = "https://www.googleapis.com/youtube/v3"


class YouTubeError(RuntimeError):
    pass


def _get(endpoint: str, params: dict) -> dict:
    if not settings.youtube_api_key:
        raise YouTubeError("YOUTUBE_API_KEY is not set in .env")
    r = requests.get(f"{API}/{endpoint}", params={**params, "key": settings.youtube_api_key}, timeout=30)
    if r.status_code == 403 and "commentsDisabled" in r.text:
        return {"items": []}
    if r.status_code != 200:
        raise YouTubeError(f"{endpoint} failed ({r.status_code}): {r.text[:300]}")
    return r.json()


def search_videos(query: str, max_results: int = 5) -> list[dict]:
    data = _get("search", {
        "part": "snippet", "q": query, "type": "video", "maxResults": max_results,
        "regionCode": "IN", "relevanceLanguage": "en", "order": "relevance",
    })
    return [{"video_id": it["id"]["videoId"], "title": it["snippet"]["title"]} for it in data.get("items", [])]


def video_titles(video_ids: list[str]) -> dict[str, str]:
    if not video_ids:
        return {}
    data = _get("videos", {"part": "snippet", "id": ",".join(video_ids[:50])})
    return {it["id"]: it["snippet"]["title"] for it in data.get("items", [])}


def parse_comment_threads(payload: dict, video_id: str, video_title: str, product_id: str) -> list[dict]:
    """Turn one commentThreads.list page into raw_reviews rows (top-level comments + inline replies)."""
    out = []
    for thread in payload.get("items", []):
        comments = [thread["snippet"]["topLevelComment"]]
        comments += thread.get("replies", {}).get("comments", [])
        for c in comments:
            s = c["snippet"]
            author = (s.get("authorChannelId") or {}).get("value") or s.get("authorDisplayName")
            out.append({
                "review_id": f"yt:{c['id']}",
                "source": "youtube",
                "source_url": f"https://www.youtube.com/watch?v={video_id}&lc={c['id']}",
                "product_id": product_id,
                "text": s.get("textOriginal") or s.get("textDisplay") or "",
                "author_hash": author_hash(author),
                "created_at": parse_ts(s.get("publishedAt")),
                "likes": s.get("likeCount", 0),
                "parent_context": video_title,
            })
    return out


def fetch_comments(video_id: str, video_title: str, product_id: str, max_comments: int = 500) -> list[dict]:
    rows, token = [], None
    while len(rows) < max_comments:
        params = {"part": "snippet,replies", "videoId": video_id, "maxResults": 100,
                  "textFormat": "plainText", "order": "relevance"}
        if token:
            params["pageToken"] = token
        payload = _get("commentThreads", params)
        rows += parse_comment_threads(payload, video_id, video_title, product_id)
        token = payload.get("nextPageToken")
        if not token:
            break
    return rows[:max_comments]


def collect(con, products: list[dict], videos_per_query: int = 4, max_comments: int = 500) -> dict:
    stats = {}
    for p in products:
        cfg = p.get("youtube", {})
        videos = dict(video_titles(cfg.get("video_ids", [])))
        for q in cfg.get("search_queries", []):
            for v in search_videos(q, videos_per_query):
                videos.setdefault(v["video_id"], v["title"])
        added = 0
        for vid, title in videos.items():
            try:
                added += insert_raw(con, fetch_comments(vid, title, p["id"], max_comments))
            except YouTubeError as e:
                log.warning("skipping video %s: %s", vid, e)
        stats[p["id"]] = {"videos": len(videos), "comments_added": added}
        log.info("youtube %s: %s", p["id"], stats[p["id"]])
    return stats
