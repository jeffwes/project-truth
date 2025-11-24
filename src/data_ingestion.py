"""Data ingestion helpers: URL scraping, YouTube transcript, and passthrough.

Functions return a single cleaned string of text.
"""
from __future__ import annotations

import re
from typing import Optional

import requests


def fetch_from_url(url: str, timeout: int = 15) -> str:
    """Fetch main text content from a URL and return as a cleaned string.

    This is a lightweight scraper: it prefers <article> or <main>, else
    falls back to concatenating <p> text. Basic error handling for bad URLs.
    """
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        return f"[ERROR] network error: {e}"

    if resp.status_code != 200:
        return f"[ERROR] HTTP {resp.status_code}: {resp.text[:200]}"

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return "[ERROR] BeautifulSoup (bs4) not installed"

    # try common containers
    main = soup.find("article") or soup.find("main")
    if main:
        texts = [p.get_text(separator=" ").strip() for p in main.find_all("p")]
        joined = "\n\n".join([t for t in texts if t])
        if joined:
            return _clean_text(joined)

    # fallback: gather large <p> blocks
    paragraphs = [p.get_text(separator=" ").strip() for p in soup.find_all("p")]
    if paragraphs:
        return _clean_text("\n\n".join(paragraphs))

    # last resort: return page title and meta description
    title = soup.title.string if soup.title else ""
    desc = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        desc = meta.get("content")
    if title or desc:
        return _clean_text(f"{title}\n\n{desc}")

    return ""


def fetch_from_youtube(url: str) -> str:
    """Fetch the transcript for a YouTube URL and return concatenated text.

    Uses youtube-transcript-api. Returns an error message string on failure.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        return "[ERROR] youtube_transcript_api package not installed"

    # extract video id
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", url)
    if not m:
        return "[ERROR] could not parse YouTube video id from URL"
    vid = m.group(1)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid)
        texts = [seg.get("text", "") for seg in transcript]
        return _clean_text("\n".join(texts))
    except Exception as e:
        return f"[ERROR] could not fetch transcript: {e}"


def passthrough_text(text: str) -> str:
    return _clean_text(text or "")


def _clean_text(s: str) -> str:
    s = s.strip()
    # collapse repeated whitespace
    s = re.sub(r"\s+", " ", s)
    return s
