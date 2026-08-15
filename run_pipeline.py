#!/usr/bin/env python3
"""
Runs the full pipeline once: fetch feeds -> cluster -> label each story
verified (2+ independently owned outlets) or single-source -> skip ones
already published -> AI-write the rest, with a fetched source image ->
store in SQLite.

Run manually:
    python run_pipeline.py

Run on a schedule (e.g. every 30 min) with cron:
    */30 * * * * cd /path/to/newsapp && /path/to/venv/bin/python run_pipeline.py >> pipeline.log 2>&1

Note: the lock file below only protects against overlapping runs on the
SAME machine. If you ever run this on multiple machines against a
shared DB, replace it with a DB-based lease instead.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import db
from pipeline.cluster_news import get_clusters
from pipeline.ai_writer import write_article
from pipeline.images import get_cluster_image

APP_DIR = Path(__file__).resolve().parent
LOCK_PATH = APP_DIR / ".pipeline.lock"

# Single-source stories are no longer filtered out, but every RSS item
# that doesn't cluster with another becomes its own single-source
# "story" -- across 11 feeds x 25 items that can be 100+ per run. This
# caps how many single-source stories get AI-written per run so one
# pipeline tick can't quietly burn through a large number of API calls.
# Cross-verified stories are never capped. Override via env if needed.
MAX_SINGLE_SOURCE_PER_RUN = int(os.environ.get("MAX_SINGLE_SOURCE_PER_RUN", "15"))


def acquire_lock():
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock():
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def run():
    """Raises RuntimeError on a missing key rather than calling sys.exit,
    so this is safe to call from inside a long-running process (e.g. the
    web app's background scheduler) without taking the whole process
    down. The __main__ block below converts that back into an exit code
    for CLI use."""
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Add it to your .env file or "
            "export it before running."
        )

    db.init_db()

    print("Fetching + clustering feeds...")
    all_clusters = get_clusters()
    verified = [c for c in all_clusters if c["verified"]]
    singles = [c for c in all_clusters if not c["verified"]][:MAX_SINGLE_SOURCE_PER_RUN]
    clusters = verified + singles
    print(f"Found {len(verified)} cross-verified + {len(singles)} single-source "
          f"stories to consider (single-source capped at {MAX_SINGLE_SOURCE_PER_RUN}/run).\n")

    written, skipped, failed = 0, 0, 0
    for cluster in clusters:
        chash = db.cluster_hash(cluster)
        if db.article_exists(chash):
            skipped += 1
            continue

        tag = "verified" if cluster["verified"] else "single-source"
        top_titles = ", ".join(it["title"][:40] for it in cluster["items"][:2])
        print(f"[{cluster['source_count']} sources, {tag}] Writing: {top_titles}...")

        result = write_article(cluster)
        if result is None:
            failed += 1
            continue

        image_url, image_credit = get_cluster_image(cluster)
        if image_url:
            print(f"  -> image from {image_credit}")

        slug = db.save_article(cluster, result, image_url, image_credit)
        if slug is None:
            # Another run (or another pass this run) already saved this
            # exact story between our exists-check and our insert -- not
            # an error, just a race we lost harmlessly.
            print("  -> already saved by a concurrent run, skipping")
            skipped += 1
            continue

        print(f"  -> saved as /article/{slug}")
        written += 1
        time.sleep(1)  # light pacing between API calls

    print(f"\nDone. {written} new articles written, {skipped} already existed, "
          f"{failed} failed to generate.")


if __name__ == "__main__":
    if not acquire_lock():
        print(f"Another run appears to be in progress (lock file exists at "
              f"{LOCK_PATH}). If you're sure that's stale, delete it and retry.")
        sys.exit(1)
    try:
        run()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        release_lock()
