import feedparser
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# ------------------------------------------------------------------ config

RSS_FEEDS = {
    "onlinekhabar": "https://www.onlinekhabar.com/feed",
}

# ------------------------------------------------------------------ schema

@dataclass
class Article:
    source:    str
    url:       str
    headline:  str
    body:      str
    image:     Optional[str]
    published: Optional[str]


# ------------------------------------------------------------------ helpers

def _parse_date(entry) -> Optional[str]:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6]).isoformat()
    return None


def _extract_body(entry) -> str:
    """
    Pull body text from the RSS entry.

    Onlinekhabar (WordPress) includes the full article inside
    <content:encoded> — no page scraping needed.

    Priority:
      1. content:encoded  (full article HTML)
      2. description/summary (shorter excerpt)
    """
    # feedparser maps <content:encoded> -> entry.content list
    content_list = entry.get("content", [])
    if content_list:
        html = content_list[0].get("value", "")
        if html:
            return BeautifulSoup(html, "html.parser").get_text(
                separator="\n", strip=True
            )

    # fallback: <description> is already stripped of tags by feedparser
    summary = entry.get("summary", "")
    if summary:
        return BeautifulSoup(summary, "html.parser").get_text(
            separator="\n", strip=True
        )

    return ""


def _extract_image(entry) -> Optional[str]:
    """
    Onlinekhabar puts a bare <image> tag inside each <item>.
    feedparser surfaces it under entry.get("image") or
    entry.get("onlinekhabar_image") depending on version.
    Fall back to the first enclosure.
    """
    img = entry.get("image")
    if isinstance(img, str) and img.startswith("http"):
        return img
    if isinstance(img, dict):
        return img.get("href") or img.get("url")
    # try media thumbnail
    media = entry.get("media_thumbnail", [])
    if media:
        return media[0].get("url")
    # try enclosures
    for enc in entry.get("enclosures", []):
        if enc.get("type", "").startswith("image"):
            return enc.get("href")
    return None


# ------------------------------------------------------------------ public

def fetch_feed(source: str, limit: int = 20) -> list[Article]:
    """
    Parse the RSS feed for the given source and return articles
    with headline + full body extracted from <content:encoded>.

    Synchronous — feedparser handles the HTTP request internally.
    No additional scraping required.

    Args:
        source:  key in RSS_FEEDS  e.g. "onlinekhabar"
        limit:   max articles to return

    Returns:
        list[Article]
    """
    feed_url = RSS_FEEDS.get(source)
    if not feed_url:
        raise ValueError(
            f"Unknown source '{source}'. Available: {list(RSS_FEEDS.keys())}"
        )

    feed = feedparser.parse(feed_url)

    if feed.bozo and not feed.entries:
        raise RuntimeError(
            f"Failed to parse feed for '{source}': {feed.bozo_exception}"
        )

    articles = []
    for entry in feed.entries[:limit]:
        url      = entry.get("link", "").strip()
        headline = entry.get("title", "").strip()

        if not url or not headline:
            continue

        articles.append(Article(
            source=source,
            url=url,
            headline=headline,
            body=_extract_body(entry),
            image=_extract_image(entry),
            published=_parse_date(entry),
        ))

    return articles
