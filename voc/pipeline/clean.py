"""Stage 1: scrub PII, tag language (en / hi / hinglish), drop junk and duplicates."""
from __future__ import annotations

import hashlib
import re

from voc.db import run_step

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
MENTION_RE = re.compile(r"(?<!\w)@[\w.]{2,}")
WS_RE = re.compile(r"\s+")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Common romanised Hindi function words. Two or more hits => Hinglish.
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nhi", "bahut", "bhut", "bohot", "accha", "acha", "achha", "kya", "mera", "meri",
    "mere", "mujhe", "kuch", "bilkul", "lekin", "aur", "bhi", "ho", "gaya", "gayi", "raha", "rahi", "tha",
    "thi", "kar", "karta", "karti", "lagta", "lagti", "yeh", "ye", "woh", "wo", "isse", "isko", "sab",
    "ekdum", "zyada", "jyada", "kam", "bekar", "bakwas", "paisa", "paise", "baal", "chehra", "dane",
}

MIN_WORDS = 4
MAX_CHARS = 2000


def scrub(text: str) -> str:
    text = URL_RE.sub("[link]", text)
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub("[phone]", text)
    text = MENTION_RE.sub("@user", text)
    return WS_RE.sub(" ", text).strip()[:MAX_CHARS]


def detect_lang(text: str) -> str:
    if len(DEVANAGARI_RE.findall(text)) >= 5:
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(1 for w in words if w in HINGLISH_MARKERS)
    return "hinglish" if hits >= 2 else "en"


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0900-\u097f]", "", text.lower())


def run(con) -> dict:
    with run_step(con, "clean") as (_, stats):
        raw = con.execute("""
            SELECT r.review_id, r.product_id, r.source, r.source_url, r.text, r.created_at, r.likes
            FROM raw_reviews r LEFT JOIN reviews c USING (review_id)
            WHERE c.review_id IS NULL
            ORDER BY r.likes DESC NULLS LAST
        """).fetchall()

        # Seed the dedupe sets with what's already clean, so re-runs stay idempotent.
        exact, prefix = set(), set()
        for pid, text in con.execute("SELECT product_id, text FROM reviews").fetchall():
            k = norm_key(text)
            exact.add((pid, hashlib.md5(k.encode()).hexdigest()))
            prefix.add((pid, k[:120]))

        keep, dropped = [], {"too_short": 0, "duplicate": 0}
        for rid, pid, source, url, text, created_at, likes in raw:
            clean = scrub(text or "")
            if len(clean.split()) < MIN_WORDS:
                dropped["too_short"] += 1
                continue
            k = norm_key(clean)
            h = (pid, hashlib.md5(k.encode()).hexdigest())
            p = (pid, k[:120])
            if h in exact or (len(k) >= 120 and p in prefix):
                dropped["duplicate"] += 1
                continue
            exact.add(h)
            prefix.add(p)
            keep.append((rid, pid, source, url, clean, detect_lang(clean), created_at, likes))

        con.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)", keep)
        langs = {}
        for row in keep:
            langs[row[5]] = langs.get(row[5], 0) + 1
        stats.update({"raw_considered": len(raw), "kept": len(keep), "dropped": dropped, "languages": langs})
    return stats
