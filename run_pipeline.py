#!/usr/bin/env python3
"""
Runs the full pipeline once: fetch feeds -> cluster -> filter to
cross-verified stories -> skip ones already published -> AI-write the
rest -> store in SQLite.

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
from pipeline.cluster_news import get_verified_clusters
from pipeline.ai_writer import write_article

APP_DIR = Path(__file__).resolve().parent
LOCK_PATH = APP_DIR / ".pipeline.lock"


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
    clusters = get_verified_clusters()
    print(f"Found {len(clusters)} cross-verified clusters.\n")

    written, skipped, failed = 0, 0, 0
    for cluster in clusters:
        chash = db.cluster_hash(cluster)
        if db.article_exists(chash):
            skipped += 1
            continue

        top_titles = ", ".join(it["title"][:40] for it in cluster["items"][:2])
        print(f"[{cluster['source_count']} sources] Writing: {top_titles}...")

        result = write_article(cluster)
        if result is None:
            failed += 1
            continue

        slug = db.save_article(cluster, result)
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
