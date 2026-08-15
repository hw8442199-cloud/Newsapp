#!/usr/bin/env python3
"""
Article image extraction
--------------------------
Given a source article URL, fetches the page and pulls its og:image /
twitter:image meta tag -- the same image the outlet itself uses when the
article is shared on social media. No AI-generated images, no stock
photos: if none of a story's source articles expose one, the story is
shown without an image rather than a fake one.
"""

import html
import re
from urllib.parse import urljoin, urlparse

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 8
# Meta tags always live in <head>, near the top of the document -- we
# don't need the full page, just enough of it to be fast and light.
MAX_BYTES = 400_000

_META_PATTERNS = (
    re.compile(
        r'<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image)(?::src)?["\']'
        r'[^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\']'
        r'[^>]+(?:property|name)=["\'](?:og:image|twitter:image)(?::src)?["\']',
        re.IGNORECASE,
    ),
)


def _safe_image_url(raw_url, base_url):
    if not raw_url:
        return None
    raw_url = html.unescape(raw_url.strip())
    resolved = urljoin(base_url, raw_url)
    parsed = urlparse(resolved)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return resolved


def fetch_og_image(article_url):
    """Returns the article's og:image/twitter:image URL, or None if the
    page couldn't be fetched or doesn't expose one. Any failure (site
    blocks bots, times out, isn't HTML, etc.) is treated as 'no image'
    rather than raised -- this is a best-effort enrichment step, not
    something that should ever take the pipeline down."""
    try:
        resp = requests.get(article_url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type.lower():
            return None
        chunks, total = [], 0
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total >= MAX_BYTES:
                break
        resp.close()
        html_text = b"".join(chunks).decode("utf-8", errors="ignore")
    except requests.RequestException:
        return None

    for pattern in _META_PATTERNS:
        match = pattern.search(html_text)
        if match:
            image_url = _safe_image_url(match.group(1), article_url)
            if image_url:
                return image_url
    return None


def get_cluster_image(cluster):
    """Tries each source item's article page (most recent first) until
    one yields a usable image. Returns (image_url, credit_source), or
    (None, None) if nothing was found across every source -- callers
    should just render without an image in that case."""
    items = sorted(cluster["items"], key=lambda it: it.get("published_at", ""), reverse=True)
    seen_links = set()
    for item in items:
        link = item.get("link")
        if not link or link in seen_links:
            continue
        seen_links.add(link)
        image_url = fetch_og_image(link)
        if image_url:
            return image_url, item["source"]
    return None, None
