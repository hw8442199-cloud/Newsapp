#!/usr/bin/env python3
"""
News Story Clustering + Cross-Verification
--------------------------------------------
Pulls items from RSS feeds, groups items covering the same real-world
story (TF-IDF + cosine similarity, with title weighting + a time-window
and shared-vocabulary gate to cut down on false matches), then returns
only clusters that are cross-verified by 2+ *independently owned*
outlets. Single-source stories are left out entirely -- see
get_verified_clusters().

Install:
    pip install feedparser requests scikit-learn numpy
"""

import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, urlunparse

import feedparser
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FEEDS = {
    "Business Standard":      "https://www.business-standard.com/rss/latest.rss",
    "BusinessLine":           "https://www.thehindubusinessline.com/feeder/default.rss",
    "Hindustan Times":        "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    "Mint News":              "https://www.livemint.com/rss/news",
    "Mint Markets":           "https://www.livemint.com/rss/markets",
    "NDTV":                   "https://feeds.feedburner.com/NDTV-LatestNews",
    "News Inshorts":          "https://inshorts.com/en/rss",
    "News18":                 "https://www.news18.com/rss/india.xml",
    "Republic":               "https://www.republicworld.com/rss/india.xml",
    "The New Indian Express": "https://www.newindianexpress.com/feed",
    "Times of India":         "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
}

# Outlets that share a newsroom/ownership group don't count as independent
# corroboration of each other. This list is illustrative, not authoritative
# -- ownership changes hands, so verify and maintain it yourself rather than
# trusting it blindly. Anything not listed here is assumed independent.
SOURCE_GROUPS = {
    "Hindustan Times": "ht-media",
    "Mint News":       "ht-media",
    "Mint Markets":    "ht-media",
    "News18":          "network18",
}

# Factual, neutral descriptors -- ownership/format, not a trust or bias
# score. Used to (a) give the AI writer grounded input for the sourcing
# note it writes on single-source stories, and (b) show under "Sourcing"
# on the article page. Keep these to verifiable facts only.
SOURCE_PROFILES = {
    "Business Standard":      "national business daily",
    "BusinessLine":           "national business daily, published by The Hindu Group",
    "Hindustan Times":        "national English-language daily, published by HT Media",
    "Mint News":              "national business daily, published by HT Media",
    "Mint Markets":           "national business daily, published by HT Media",
    "NDTV":                   "national broadcaster and news website",
    "News Inshorts":          "news aggregator that republishes summarized versions of other outlets' reporting, rather than original reporting",
    "News18":                 "national news network, part of Network18",
    "Republic":               "national broadcaster and news website",
    "The New Indian Express": "national English-language daily",
    "Times of India":         "national English-language daily, among India's widest-circulation English newspapers",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 10

# Similarity threshold for "same story" on TF-IDF cosine similarity.
SIMILARITY_THRESHOLD = 0.40

# A cluster only counts as cross-verified once this many distinct
# ownership groups are in it (not just distinct display names -- see
# SOURCE_GROUPS).
MIN_SOURCES_FOR_VERIFIED = 2

# Drop feed items older than this -- an old item re-served by a feed
# shouldn't be treated as breaking/current.
MAX_ITEM_AGE = timedelta(hours=48)

# Two items can only be merged into the same story if their published
# times are within this window of each other, in addition to passing
# the similarity threshold. Cuts down on "same generic topic, different
# day" false matches.
TIME_WINDOW = timedelta(hours=36)

# Extra gate alongside cosine similarity: require at least this many
# shared significant (len > 3) words between two titles. Cosine
# similarity alone can be fooled by generic shared vocabulary (e.g.
# "India", "government", "Supreme Court").
MIN_SHARED_TITLE_TERMS = 2

_BOILERPLATE_SUFFIX = re.compile(
    r"\s*\|\s*(?:[^|]*\s*\|\s*)?(News18|Firstpost)\s*$",
    re.IGNORECASE,
)


def strip_boilerplate(title):
    """Drop known trailing outlet boilerplate, e.g. '... | Defence | News18'
    or '... | #plainspeak | News18'. Only strips titles matching a known
    boilerplate pattern -- a legitimate title containing '|' elsewhere is
    left alone. The full original title is still used for display."""
    return _BOILERPLATE_SUFFIX.sub("", title).strip()


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text or "")       # strip HTML
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def safe_url(url):
    """Only allow http/https links through -- an RSS feed is untrusted
    input and could in principle hand back a javascript: or data: URL."""
    try:
        parsed = urlparse(url or "")
    except ValueError:
        return None
    return url if parsed.scheme in ("http", "https") else None


def _entry_published_at(entry):
    struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if not struct:
        return None
    return datetime(*struct[:6], tzinfo=timezone.utc)


def fetch_items():
    items = []
    now = datetime.now(timezone.utc)
    for source, url in FEEDS.items():
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code != 200:
                print(f"  [skip] {source}: HTTP {resp.status_code}")
                continue
            parsed = feedparser.parse(resp.content)
            for entry in parsed.entries[:25]:
                published_at = _entry_published_at(entry)
                if published_at is None:
                    continue  # no timestamp -- can't confirm it's current
                if now - published_at > MAX_ITEM_AGE:
                    continue  # stale, feed is re-serving an old item

                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                link = safe_url(getattr(entry, "link", ""))
                if not link or not title.strip():
                    continue

                items.append({
                    "source": source,
                    "title": title.strip(),
                    "summary": summary.strip(),
                    "link": link,
                    "published_at": published_at.isoformat(),
                    "text": clean_text(strip_boilerplate(title) + " " + summary),
                })
        except Exception as e:
            print(f"  [skip] {source}: {type(e).__name__}")
    return items


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _weighted_text(item):
    """Title terms count for more than summary terms -- the headline is
    the strongest signal for 'is this the same story'."""
    title_clean = clean_text(strip_boilerplate(item["title"]))
    return f"{title_clean} {title_clean} {item['text']}"


def _same_story_gate(a, b):
    """Extra check beyond cosine similarity, so two stories sharing only
    generic vocabulary don't get merged: published close together in
    time AND sharing real title vocabulary, not just topic words."""
    try:
        ta = datetime.fromisoformat(a["published_at"])
        tb = datetime.fromisoformat(b["published_at"])
    except (KeyError, ValueError):
        return False
    if abs((ta - tb).total_seconds()) > TIME_WINDOW.total_seconds():
        return False

    terms_a = {w for w in clean_text(strip_boilerplate(a["title"])).split() if len(w) > 3}
    terms_b = {w for w in clean_text(strip_boilerplate(b["title"])).split() if len(w) > 3}
    return len(terms_a & terms_b) >= MIN_SHARED_TITLE_TERMS


def cluster_items(items):
    """Groups items into stories using deterministic connected-components
    clustering (union-find) rather than a greedy order-dependent pass, so
    the result doesn't depend on feed fetch order. Returns a list of
    groups, each a list of item dicts."""
    items = [it for it in items if it["text"].strip()]
    if len(items) < 2:
        return []

    texts = [_weighted_text(it) for it in items]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # e.g. every text was stripped down to nothing (all-stopword titles)
        return []

    sim = cosine_similarity(matrix)
    n = len(items)
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] >= SIMILARITY_THRESHOLD and _same_story_gate(items[i], items[j]):
                uf.union(i, j)

    groups = defaultdict(list)
    for idx in range(n):
        groups[uf.find(idx)].append(items[idx])
    return list(groups.values())


def get_clusters():
    """Fetch, cluster, and return EVERY story -- cross-verified and
    single-source alike -- as a list of dicts shaped for the rest of the
    pipeline: {items, sources, source_count, owner_group_count, verified}.

    Nothing is dropped here. `verified` (2+ independently owned outlets,
    see SOURCE_GROUPS) is a label the rest of the pipeline and the UI use
    to tell the reader how corroborated a story is -- single-source
    stories still get published, just marked as such. Cross-verified
    clusters sort first; within each tier, most recent lead item first."""
    items = fetch_items()
    groups = cluster_items(items)

    clusters = []
    for group in groups:
        sources = sorted({it["source"] for it in group})
        owner_groups = {SOURCE_GROUPS.get(s, s) for s in sources}
        clusters.append({
            "items": group,
            "sources": sources,
            "source_count": len(sources),
            "owner_group_count": len(owner_groups),
            "verified": len(owner_groups) >= MIN_SOURCES_FOR_VERIFIED,
        })

    def _lead_time(c):
        return max((it.get("published_at", "") for it in c["items"]), default="")

    clusters.sort(key=lambda c: (c["verified"], c["source_count"], _lead_time(c)), reverse=True)
    return clusters


# Older name -- used to return only the verified subset. Now returns
# everything (see get_clusters docstring). Kept as an alias so nothing
# importing the old name breaks; prefer get_clusters() going forward.
get_verified_clusters = get_clusters


def main():
    print(f"Fetching from {len(FEEDS)} feeds...\n")
    clusters = get_clusters()
    verified = [c for c in clusters if c["verified"]]
    singles = [c for c in clusters if not c["verified"]]
    print("=" * 78)
    print(f"{len(clusters)} STORIES  ({len(verified)} cross-verified, {len(singles)} single-source)")
    print("=" * 78)
    for c in clusters:
        tag = "VERIFIED" if c["verified"] else "single-source"
        print(f"\n[{c['source_count']} sources, {tag}: {', '.join(c['sources'])}]")
        for it in c["items"]:
            print(f"  - ({it['source']}) {it['title']}")


if __name__ == "__main__":
    main()
