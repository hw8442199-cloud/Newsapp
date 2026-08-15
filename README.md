# Wire & Brass — AI newsroom pipeline

Fetches RSS feeds, clusters items into stories, and writes a full
article via NVIDIA Nemotron for every story -- cross-verified (2+
*independently owned* outlets) and single-source alike. Single-source
stories are clearly labeled and hedged in the writing; cross-verified
ones aren't. Each article also carries the real image from one of its
source articles (og:image), when one is available. Serves the result as
a site + your own RSS feed.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in NVIDIA_API_KEY and SITE_URL
```

On Replit, use **Secrets** instead of a committed `.env` file.

## Running it

```bash
# One-off: fetch, cluster, write up anything new
python run_pipeline.py

# The site (also runs the pipeline automatically in the background --
# see "Scheduling" below)
uvicorn main:app --reload
```

## Scheduling

`main.py` runs the pipeline itself, as a background task inside the
same process, every `PIPELINE_INTERVAL_SECONDS` (default 1800 = 30
min) -- see the top of `main.py`. This keeps things to a single
deployable service with one shared disk, instead of a separate cron
service that would need its own (unshared) storage.

Trade-off: if your host spins the service down after inactivity (e.g.
Render's free tier, after ~15 min with no requests), this loop pauses
along with everything else and resumes on the next request. Fine for a
low-traffic hobby site. If you need guaranteed on-schedule runs
regardless of traffic, either move to an always-on plan, or set
`ENABLE_BACKGROUND_PIPELINE=false` and drive `run_pipeline.py` from an
external scheduler instead (in which case give it its own persistent
disk/DB, since it won't be sharing the web service's).

## Deploying to Render

A `render.yaml` is included. In the Render dashboard: **New +** ->
**Blueprint**, point it at your GitHub repo, and it'll create the web
service from that file. After the first deploy, set `NVIDIA_API_KEY`
as a secret in the service's Environment tab (Blueprints deliberately
don't let you commit secret values), and double check `SITE_URL`
matches your actual `*.onrender.com` URL (or your custom domain).

The `disk:` block in `render.yaml` adds a small persistent disk
(~$0.25/mo) so the SQLite file survives redeploys. Delete that block
(and the `NEWS_DB_PATH` env var) to stay on the fully free plan --
just know the database may not survive every redeploy without it.

Netlify won't work for this project -- it only hosts static sites and
short-lived serverless functions, not a persistent server with a
SQLite file on disk. If you see Netlify's generic 404 after deploying
there, that's why: nothing about your app ever actually ran.

## Sourcing model

Every story that clears clustering gets written and published -- nothing
is silently dropped anymore. What changes is how it's labeled and
written:

- **Cross-verified** (2+ independently owned outlets, per `SOURCE_GROUPS`
  in `cluster_news.py`): facts are stated plainly.
- **Single-source**: the article attributes claims to the reporting
  outlet throughout ("X reports...") instead of stating them as settled
  fact, and carries a `reliability_note` -- one sentence the AI writes,
  but *grounded only in facts you gave it*: the outlet's ownership/format
  (`SOURCE_PROFILES` in `cluster_news.py`) and whether the story is
  corroborated. It's not a free-floating AI trust score -- deliberately
  so, since an LLM guessing at outlet trustworthiness from nothing is
  exactly the kind of ungrounded claim that shouldn't end up on a news
  site. `SOURCE_PROFILES` is factual (ownership, aggregator vs. original
  reporting) and, like `SOURCE_GROUPS`, illustrative rather than
  authoritative -- maintain it yourself.

Single-source volume can be large (every unclustered RSS item becomes
its own story), so `run_pipeline.py` caps how many single-source stories
get AI-written per run via `MAX_SINGLE_SOURCE_PER_RUN` (default 15;
override via env var). Cross-verified stories are never capped.

## Images

`pipeline/images.py` pulls the `og:image` (falling back to
`twitter:image`) from one of a story's source articles -- the same image
the outlet itself uses when the article is shared on social media. If no
source exposes one, the story is shown without an image. No AI-generated
images, no stock photos.

## What changed from the reviewed version

Fixed per the review, grouped by what they actually affect:

**Wouldn't run at all**
- `pipeline/` is now a real package with `get_verified_clusters()`
  actually defined and returning the shape `run_pipeline.py` expects.
- `requirements.txt` added.
- NVIDIA client is created lazily (`get_client()`), so importing the
  module doesn't crash before your own "is the key set" check runs.
- `main.py` and `db.py` use paths based on `Path(__file__)`, not the
  current working directory — the DB and templates won't silently move
  or 404 depending on where you launch from.

**Would silently produce wrong results**
- Clustering is now deterministic connected-components (union-find)
  instead of a greedy pass — feed fetch order no longer changes which
  stories get grouped together.
- Verification counts *ownership groups* (`SOURCE_GROUPS`), not just
  outlet names — two outlets under the same media house no longer
  count as independent corroboration of each other. **The group
  mapping in `cluster_news.py` is illustrative, not authoritative** —
  ownership changes, so verify and maintain it yourself.
- Similarity is gated by a time window + shared title vocabulary on top
  of cosine similarity, so two unrelated stories that just share
  generic words ("India", "government") are less likely to merge.
  Full entity/date/location extraction (real NER) would be more
  rigorous still — that's a reasonable next step, not implemented here.
- `strip_boilerplate()` only strips known trailing-tag patterns now,
  so a legitimate title containing `|` isn't truncated.
- Feed items with no publish timestamp, or older than 48h, are dropped
  before clustering.
- `db.save_article()` uses `ON CONFLICT ... DO NOTHING` and checks
  `cursor.rowcount`, returning `None` when nothing was actually
  inserted — `run_pipeline.py` now treats that as "already exists"
  instead of printing a false "saved" message.
- `cluster_hash()` normalizes links (drops tracking params, trailing
  slashes) and folds in source names + lead title, so the same story
  reached via a slightly different URL doesn't get republished.

**Would break under real usage**
- `write_article()` retries on failure (exponential backoff, 3
  attempts) instead of taking down the whole pipeline run on one bad
  API call.
- Prompts are capped: deduped by link, max 6 items per cluster, 1500
  chars per summary.
- Model output is validated (all three fields present and non-empty
  strings) before use, instead of trusting `.strip()` on whatever came
  back.
- SQLite runs in WAL mode with a busy timeout, since the web server and
  the pipeline both touch the same file.
- `run_pipeline.py` takes a simple lock file so two overlapping runs
  (e.g. a slow run plus the next cron tick) don't both call the AI API
  for the same story. This only covers one machine — if you ever run
  the pipeline from more than one host against a shared DB, replace it
  with a DB-based lease.
- RSS/source links are validated to `http(s)` only (`safe_url()`)
  before they're used, since RSS content is untrusted input.
- The AI prompt explicitly frames source material as untrusted data and
  instructs the model not to follow anything embedded in it.

**Deployment correctness**
- `SITE_URL` prints a clear warning if unset (RSS links would otherwise
  point at `localhost`) — set it before you actually deploy.
