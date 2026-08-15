#!/usr/bin/env python3
"""
SQLite storage for generated articles.
"""

import os
import re
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse

APP_DIR = Path(__file__).resolve().parent
# Absolute by default so the DB lands in the same place no matter what
# directory the server/pipeline is started from. Override with
# NEWS_DB_PATH if you want it somewhere else (e.g. a persistent volume).
DB_PATH = os.environ.get("NEWS_DB_PATH", str(APP_DIR / "news.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT UNIQUE NOT NULL,
    headline TEXT NOT NULL,
    dek TEXT NOT NULL,
    body TEXT NOT NULL,
    sources TEXT NOT NULL,        -- JSON array of outlet names
    source_count INTEGER NOT NULL,
    source_links TEXT NOT NULL,   -- JSON array of {source, title, link}
    cluster_hash TEXT UNIQUE NOT NULL,  -- dedupe key so re-runs don't republish
    published_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_published_at ON articles (published_at DESC);
"""


@contextmanager
def get_conn():
    # timeout + WAL + busy_timeout: the API server (reads) and the
    # pipeline (writes) can hit this file at the same time.
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def slugify(headline, chash):
    base = re.sub(r"[^a-z0-9]+", "-", headline.lower()).strip("-")[:70]
    return f"{base}-{chash[:8]}"


def _normalize_link(url):
    """Strip tracking params/fragments and trailing slashes so the same
    article reached via slightly different URLs (utm params, AMP path,
    etc.) still hashes the same."""
    try:
        p = urlparse(url)
    except ValueError:
        return url
    path = p.path.rstrip("/")
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", "", ""))


def cluster_hash(cluster):
    """Stable fingerprint for a story: normalized source links PLUS
    sorted source names and the lead (earliest) item's cleaned title, so
    a slightly different re-clustering of the same story -- or a link
    with different tracking params on a later run -- still lands on the
    same hash instead of getting republished."""
    items = cluster["items"]
    links = sorted({_normalize_link(it["link"]) for it in items if it.get("link")})
    sources = sorted({it["source"] for it in items})
    lead_title = min(items, key=lambda it: it.get("published_at", ""))["title"]
    lead_title_norm = re.sub(r"[^a-z0-9]+", " ", lead_title.lower()).strip()

    raw = "|".join(links) + "||" + "|".join(sources) + "||" + lead_title_norm
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def article_exists(chash):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM articles WHERE cluster_hash = ?", (chash,)
        ).fetchone()
        return row is not None


def save_article(cluster, written):
    """Returns the new article's slug, or None if nothing was actually
    inserted (e.g. a concurrent run already saved this exact story --
    see the ON CONFLICT clause). Callers must check for None rather than
    assuming a slug back means a row was written."""
    chash = cluster_hash(cluster)
    slug = slugify(written["headline"], chash)
    source_links = [
        {"source": it["source"], "title": it["title"], "link": it["link"]}
        for it in cluster["items"]
    ]
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO articles
               (slug, headline, dek, body, sources, source_count,
                source_links, cluster_hash, published_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(cluster_hash) DO NOTHING""",
            (
                slug,
                written["headline"],
                written["dek"],
                written["body"],
                json.dumps(cluster["sources"]),
                cluster["source_count"],
                json.dumps(source_links),
                chash,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        if cur.rowcount == 0:
            return None
    return slug


def list_articles(limit=50, offset=0):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM articles ORDER BY published_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_article(slug):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM articles WHERE slug = ?", (slug,)
        ).fetchone()
        return _row_to_dict(row) if row else None


def _row_to_dict(row):
    d = dict(row)
    d["sources"] = json.loads(d["sources"])
    d["source_links"] = json.loads(d["source_links"])
    return d
