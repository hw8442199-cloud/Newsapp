#!/usr/bin/env python3
"""
FastAPI site serving the AI-written, cross-verified articles.

Run:
    uvicorn main:app --reload
"""

import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from email.utils import format_datetime
from xml.sax.saxutils import escape

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import db
import run_pipeline

APP_DIR = Path(__file__).resolve().parent

# How often the background scheduler below fetches + writes new stories.
# Runs inside this same process rather than as a separate host/cron
# service, so it shares this service's disk (and therefore the SQLite
# file) with no extra moving parts. NOTE: on a platform that spins the
# service down after inactivity (e.g. Render's free tier), this loop
# pauses whenever the service is asleep and resumes on the next request
# -- fine for a low-traffic site, but if you need guaranteed on-schedule
# runs, move to an always-on plan or an external cron hitting a trigger
# endpoint instead.
PIPELINE_INTERVAL_SECONDS = int(os.environ.get("PIPELINE_INTERVAL_SECONDS", "1800"))
ENABLE_BACKGROUND_PIPELINE = os.environ.get("ENABLE_BACKGROUND_PIPELINE", "true").lower() == "true"


async def _pipeline_loop():
    while True:
        if os.environ.get("NVIDIA_API_KEY"):
            if run_pipeline.acquire_lock():
                try:
                    await asyncio.to_thread(run_pipeline.run)
                except Exception as e:
                    print(f"[scheduler] pipeline run failed: {type(e).__name__}: {e}")
                finally:
                    run_pipeline.release_lock()
            else:
                print("[scheduler] a pipeline run is already in progress elsewhere, skipping this tick")
        else:
            print("[scheduler] NVIDIA_API_KEY not set, skipping this tick")
        await asyncio.sleep(PIPELINE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_pipeline_loop()) if ENABLE_BACKGROUND_PIPELINE else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Wire & Brass", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

SITE_NAME = "Wire & Brass"
SITE_TAGLINE = "News, cross-checked before it's written."

_site_url_env = os.environ.get("SITE_URL")
if not _site_url_env:
    print("WARNING: SITE_URL is not set -- RSS item links will point at "
          "localhost. Set SITE_URL to your public domain before deploying.")
SITE_URL = (_site_url_env or "http://localhost:8000").rstrip("/")

db.init_db()


def _fmt_date(iso_string):
    dt = datetime.fromisoformat(iso_string)
    return dt.strftime("%b %-d, %Y · %-I:%M %p UTC")


templates.env.filters["fmt_date"] = _fmt_date


@app.get("/")
def home(request: Request):
    articles = db.list_articles(limit=30)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "site_name": SITE_NAME,
            "site_tagline": SITE_TAGLINE,
            "articles": articles,
        },
    )


@app.get("/article/{slug}")
def article(request: Request, slug: str):
    art = db.get_article(slug)
    if not art:
        raise HTTPException(status_code=404, detail="Article not found")
    return templates.TemplateResponse(
        request,
        "article.html",
        {
            "site_name": SITE_NAME,
            "article": art,
        },
    )


@app.get("/rss.xml")
def rss_feed():
    articles = db.list_articles(limit=50)
    items_xml = []
    for a in articles:
        link = f"{SITE_URL}/article/{a['slug']}"
        pub_date = format_datetime(datetime.fromisoformat(a["published_at"]))
        items_xml.append(f"""
    <item>
      <title>{escape(a['headline'])}</title>
      <link>{escape(link)}</link>
      <guid isPermaLink="true">{escape(link)}</guid>
      <description>{escape(a['dek'])}</description>
      <pubDate>{pub_date}</pubDate>
      <source>{escape(', '.join(a['sources']))}</source>
    </item>""")

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{escape(SITE_NAME)}</title>
    <link>{escape(SITE_URL)}</link>
    <description>{escape(SITE_TAGLINE)}</description>
    <language>en-in</language>{''.join(items_xml)}
  </channel>
</rss>"""
    return Response(content=xml, media_type="application/rss+xml")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
