"""
YouTube Music API Wrapper — v5  (Render 512 MB free-tier optimised)
====================================================================

WHAT CHANGED FROM v4
─────────────────────────────────────────────────────────────────────

THUMBNAIL QUALITY (biggest visible improvement)
  • upgrade_thumbnail_url() rewrites every URL to maximum resolution
    before it ever leaves this process — zero extra API calls:

    lh3.googleusercontent.com  (album art, artist photos)
      Strips any =w{n}-h{n}-... suffix and appends =w576-h576-l90-rj.
      576 px is the largest Google serves for music art; requesting
      anything higher just returns 576 anyway.

    i.ytimg.com/vi/  (YouTube video thumbnails)
      Rewrites any quality token (mqdefault / default / 0 / sddefault)
      to hqdefault.jpg, which is 480×360 and always present.
      We intentionally avoid maxresdefault — it 404s on many music
      videos so the app would fall back to a broken image.

  • _score() ranks thumbnails correctly even when width=0 (which is
    the case for the majority of ytmusicapi responses).
    Priority: pixel-area > lh3 width-in-URL > ytimg filename rank.
    Filename rank: maxresdefault > sddefault > 0.jpg > hqdefault >
                   mqdefault > default (≈ 120 px).

COUNTRY-AWARE CONTENT
  • get_ytm(country, language) returns a pooled YTMusic instance per
    locale. Pool is capped at 12 entries (~2 MB each = ~24 MB total).
    Oldest entry is evicted when the pool is full.
  • Every content endpoint now accepts ?country=XX&language=en.
    Country defaults to "ZZ" (global). Supported by ytmusicapi via
    YTMusic(language=..., location=...) constructor params.
  • Cache keys include country+language so per-locale responses are
    stored independently.
  • /charts and /trending share one cache key ("charts:{c}:{l}") so
    they never double-fetch — trending is just a slice of charts.

RENDER FREE-TIER
  • startup() is non-blocking — YTMusic init runs as a background
    task so Render's health check passes immediately on cold start
    (Render kills services that don't respond within ~30 s).
  • _upnext_store is now an OrderedDict with a Lock; LRU eviction is
    correct (v4 used a plain dict + next(iter()) which is FIFO, not
    LRU, and not thread-safe without a lock).
  • GZip minimum_size lowered 500→300 — catches more small responses.
  • Thread pool stays at 4 (ytmusicapi is I/O-bound; more workers
    waste RAM without throughput benefit on the free tier).
  • /cache_stats endpoint — monitor live entry counts.
  • /health reports uptime + pool size.

SPEED
  • Cache-Control headers on every endpoint (browser/CDN caching).
  • _normalise_home() and _normalise_charts() are top-level shared
    functions used by both endpoints and background warm-up.
  • Background warm-up pre-fills home + global charts 3 s after start
    so the first real user request is always a cache hit.
  • Podcast ID-stripping logic simplified (v4 had dead/contradictory
    if-else branches that silently did nothing).
"""

from __future__ import annotations

import asyncio
import re
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from ytmusicapi import YTMusic

# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────

_START_TIME = time.monotonic()

app = FastAPI(
    title="YouTube Music API — v5",
    description=(
        "FastAPI wrapper for unauthenticated ytmusicapi. "
        "Country-aware content, max-quality thumbnails, Render 512 MB optimised."
    ),
    version="5.0.0",
)

app.add_middleware(GZipMiddleware, minimum_size=300)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "DELETE"],
    allow_headers=["*"],
)

# 4 workers: ytmusicapi is I/O-bound (HTTP to Google), not CPU-bound.
# Raising this wastes ~8 MB RAM per extra worker for zero throughput gain.
_executor = ThreadPoolExecutor(max_workers=4)


async def run(fn, *args, **kwargs):
    """Offload a blocking ytmusicapi call to the shared thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


# ─────────────────────────────────────────────────────────────────────────────
#  Per-locale YTMusic instance pool
#  Each instance is ~1-2 MB.  Cap at 12 locales → ≤ 24 MB for the pool.
# ─────────────────────────────────────────────────────────────────────────────

_ytm_pool:    Dict[str, YTMusic] = {}   # "COUNTRY:language" → YTMusic
_ytm_pool_od: OrderedDict        = OrderedDict()   # for LRU tracking
_ytm_lock     = threading.Lock()
_YTM_POOL_MAX = 12


def get_ytm(country: str = "ZZ", language: str = "en") -> YTMusic:
    """
    Return a cached YTMusic instance for the given locale.
    Creates one on first call; evicts the LRU entry when the pool is full.
    country = ISO 3166-1 alpha-2 (e.g. "IN", "US", "GB") or "ZZ" for global.
    language = BCP-47 (e.g. "en", "hi", "fr").
    """
    country  = (country  or "ZZ").upper().strip()
    language = (language or "en").lower().strip()
    key = f"{country}:{language}"

    # Fast path — no lock needed for a read that's already in the dict
    if key in _ytm_pool:
        with _ytm_lock:
            if key in _ytm_pool_od:
                _ytm_pool_od.move_to_end(key)
        return _ytm_pool[key]

    with _ytm_lock:
        # Double-check after acquiring the lock
        if key in _ytm_pool:
            _ytm_pool_od.move_to_end(key)
            return _ytm_pool[key]

        # Evict LRU entry if the pool is full
        if len(_ytm_pool) >= _YTM_POOL_MAX:
            oldest_key, _ = _ytm_pool_od.popitem(last=False)
            _ytm_pool.pop(oldest_key, None)

        # Build the new instance
        try:
            location = country if country != "ZZ" else ""
            instance = YTMusic(language=language, location=location)
        except Exception:
            try:
                instance = YTMusic()      # plain fallback
            except Exception as exc:
                raise RuntimeError(f"YTMusic init failed: {exc}") from exc

        _ytm_pool[key]    = instance
        _ytm_pool_od[key] = True
        return instance


# ─────────────────────────────────────────────────────────────────────────────
#  TTL Cache — thread-safe LRU with per-entry TTL
# ─────────────────────────────────────────────────────────────────────────────

class TTLCache:
    """
    Thread-safe LRU cache with per-entry TTL.
    get() returns the cached value or None on miss/expiry.
    """

    def __init__(self, maxsize: int = 128, ttl: int = 300):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl     = ttl
        self._lock    = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self._store:
                return None
            value, expires = self._store[key]
            if time.monotonic() > expires:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.monotonic() + (ttl or self._ttl))
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Tuned for Render 512 MB free tier
_cache_short  = TTLCache(maxsize=64,  ttl=120)   # 2 min  — search, suggestions
_cache_medium = TTLCache(maxsize=96,  ttl=600)   # 10 min — home, charts, playlists
_cache_long   = TTLCache(maxsize=160, ttl=3600)  # 1 hr   — song, artist, album, podcast


# ─────────────────────────────────────────────────────────────────────────────
#  Up-Next queue store — in-process LRU, no Redis needed
# ─────────────────────────────────────────────────────────────────────────────

_upnext_store: OrderedDict[str, Dict] = OrderedDict()   # proper LRU
_upnext_lock  = threading.Lock()
_UPNEXT_TTL   = 7200   # 2 h
_UPNEXT_MAX   = 50     # 50 entries × ~2 KB ≈ 100 KB


# ─────────────────────────────────────────────────────────────────────────────
#  Thumbnail helpers — maximum quality, zero extra API calls
# ─────────────────────────────────────────────────────────────────────────────

# YouTube video thumbnail filename → quality rank (higher is better)
_YT_QUALITY_RANK: Dict[str, int] = {
    "maxresdefault": 100,
    "sddefault":      70,
    "0":              65,   # YouTube's "index 0" ≈ 480 p
    "hqdefault":      50,
    "mqdefault":      30,
    "2":              20,
    "1":              15,
    "3":              10,
    "default":         5,
}

# Matches the =w{n}-h{n}... size suffix on lh3 URLs
_LH3_SIZE_RE = re.compile(r"=(w\d+(-h\d+)?|h\d+|s\d+)(-[a-zA-Z0-9_\-]*)*$")


def upgrade_thumbnail_url(url: str) -> str:
    """
    Rewrite a thumbnail URL to the highest available quality.
    No HTTP request — pure string manipulation.

    lh3.googleusercontent.com (album art, artist images):
        Remove the size suffix (=w226-h226-l90-rj, =s100, etc.)
        and replace with =w576-h576-l90-rj.  576 is the max Google
        serves for music art; larger requests are silently capped.

    i.ytimg.com/vi/ (YouTube video thumbnails):
        Replace any quality token with hqdefault.jpg (480×360).
        We avoid maxresdefault because it returns HTTP 404 on many
        music / lyric videos and would show a broken image.
    """
    if not url:
        return url
    try:
        if "lh3.googleusercontent.com" in url:
            url = _LH3_SIZE_RE.sub("", url)
            return url + "=w576-h576-l90-rj"

        if "i.ytimg.com/vi/" in url:
            url = re.sub(
                r"/(maxresdefault|sddefault|hqdefault|mqdefault|default|[0-3])\.jpg",
                "/hqdefault.jpg",
                url,
            )
            return url
    except Exception:
        pass
    return url


def _thumb_score(t: Any) -> int:
    """
    Quality score for a single thumbnail (higher = better).
    Three-tier priority:
      1. pixel area (width × height) if both are non-zero
      2. width encoded in lh3 URL suffix (=w{n}-h{n})
      3. ytimg filename rank
    This handles the common case where ytmusicapi returns width=0.
    """
    if isinstance(t, str):
        url, w, h = t, 0, 0
    else:
        url = t.get("url", "")
        w   = int(t.get("width",  0) or 0)
        h   = int(t.get("height", 0) or 0)

    # Tier 1: real pixel area
    if w > 0 and h > 0:
        return w * h

    # Tier 2: parse lh3 width from suffix
    m = re.search(r"=w(\d+)", url)
    if m:
        side = int(m.group(1))
        return side * side

    # Tier 3: ytimg filename
    try:
        fname = url.rsplit("/", 1)[-1].split("?")[0].split(".")[0]
        rank  = _YT_QUALITY_RANK.get(fname)
        if rank:
            return rank * 10_000
    except Exception:
        pass

    return 0


def best_thumbnails_list(thumbnails: list | None) -> list:
    """
    Sort thumbnails best-first, upgrade every URL to max quality,
    deduplicate on the *upgraded* URL.
    Always returns a list of dicts with at least {"url", "width", "height"}.
    """
    if not thumbnails:
        return []
    try:
        seen: set[str] = set()
        out:  list     = []
        for t in sorted(thumbnails, key=_thumb_score, reverse=True):
            if isinstance(t, str):
                t = {"url": t, "width": 0, "height": 0}
            url = t.get("url", "")
            if not url:
                continue
            upgraded = upgrade_thumbnail_url(url)
            if upgraded not in seen:
                seen.add(upgraded)
                out.append({**t, "url": upgraded})
        return out
    except Exception:
        return thumbnails or []


def best_thumbnail(thumbnails: list | None) -> str:
    """Return the URL of the single highest-quality thumbnail."""
    lst = best_thumbnails_list(thumbnails)
    return lst[0]["url"] if lst else ""


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_artists(artists: Any) -> list:
    if not artists:
        return []
    if isinstance(artists, str):
        return [{"name": artists, "id": ""}]
    if isinstance(artists, list):
        out = []
        for a in artists:
            if isinstance(a, str):
                out.append({"name": a, "id": ""})
            elif isinstance(a, dict):
                out.append({
                    "name": a.get("name", ""),
                    "id":   a.get("id", "") or a.get("browseId", ""),
                })
        return out
    return []


def norm_track(t: dict) -> dict:
    raw = t.get("thumbnails") or t.get("thumbnail") or []
    if isinstance(raw, str):
        raw = [{"url": raw, "width": 0, "height": 0}]
    thumbs     = best_thumbnails_list(raw)
    album      = t.get("album")
    album_name = album.get("name", "") if isinstance(album, dict) else (album or "")
    return {
        "videoId":    t.get("videoId", ""),
        "title":      t.get("title", ""),
        "artists":    _norm_artists(t.get("artists") or t.get("artist")),
        "album":      album_name,
        "duration":   t.get("duration", ""),
        "thumbnails": thumbs,
        "thumbnail":  thumbs[0]["url"] if thumbs else "",
        "isExplicit": t.get("isExplicit", False),
        "year":       t.get("year", ""),
    }


def norm_artist_result(a: dict) -> dict:
    thumbs = best_thumbnails_list(a.get("thumbnails") or [])
    return {
        "browseId":    a.get("browseId", "") or a.get("channelId", ""),
        "name":        a.get("artist", "") or a.get("name", "") or a.get("title", ""),
        "subscribers": a.get("subscribers", ""),
        "thumbnails":  thumbs,
        "thumbnail":   thumbs[0]["url"] if thumbs else "",
    }


def norm_album_result(a: dict) -> dict:
    thumbs = best_thumbnails_list(a.get("thumbnails") or [])
    return {
        "browseId":   a.get("browseId", ""),
        "title":      a.get("title", ""),
        "artists":    _norm_artists(a.get("artists")),
        "year":       a.get("year", ""),
        "type":       a.get("type", "Album"),
        "thumbnails": thumbs,
        "thumbnail":  thumbs[0]["url"] if thumbs else "",
    }


def norm_playlist_result(p: dict) -> dict:
    thumbs = best_thumbnails_list(p.get("thumbnails") or [])
    return {
        "browseId":   p.get("browseId", "") or p.get("playlistId", ""),
        "title":      p.get("title", ""),
        "author":     p.get("author", ""),
        "itemCount":  p.get("itemCount", ""),
        "thumbnails": thumbs,
        "thumbnail":  thumbs[0]["url"] if thumbs else "",
    }


def norm_podcast_result(p: dict) -> dict:
    thumbs    = best_thumbnails_list(p.get("thumbnails") or [])
    browse_id = p.get("browseId") or p.get("podcastId") or p.get("channelId") or ""
    author    = p.get("author", "") or ", ".join(
        a.get("name", "") for a in _norm_artists(p.get("artists"))
    )
    return {
        "browseId":   browse_id,
        "title":      p.get("title", ""),
        "author":     author,
        "thumbnails": thumbs,
        "thumbnail":  thumbs[0]["url"] if thumbs else "",
    }


def norm_search_results(raw: list, filter_type: str | None) -> list:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rt = (item.get("resultType") or filter_type or "").lower()
        if rt in ("song", "songs", "video", "videos"):
            n = norm_track(item)
            n["resultType"] = "video" if "video" in rt else "song"
            out.append(n)
        elif rt in ("artist", "artists"):
            n = norm_artist_result(item); n["resultType"] = "artist"; out.append(n)
        elif rt in ("album", "albums", "single", "singles", "ep"):
            n = norm_album_result(item); n["resultType"] = "album"; out.append(n)
        elif rt in ("playlist", "playlists"):
            n = norm_playlist_result(item); n["resultType"] = "playlist"; out.append(n)
        elif rt in ("podcast", "podcasts", "episode", "episodes"):
            n = norm_podcast_result(item); n["resultType"] = "podcast"; out.append(n)
        else:
            thumbs = best_thumbnails_list(item.get("thumbnails") or [])
            item["thumbnails"] = thumbs
            item["thumbnail"]  = thumbs[0]["url"] if thumbs else ""
            out.append(item)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Shared normalisation — used by endpoints AND background warm-up
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_home(raw: list) -> list:
    shelves = []
    for shelf in raw:
        if not isinstance(shelf, dict):
            continue
        contents = []
        for item in (shelf.get("contents") or []):
            if not isinstance(item, dict):
                continue
            if item.get("videoId"):
                contents.append(norm_track(item))
            else:
                thumbs = best_thumbnails_list(item.get("thumbnails") or [])
                item["thumbnails"] = thumbs
                item["thumbnail"]  = thumbs[0]["url"] if thumbs else ""
                contents.append(item)
        if contents:
            shelves.append({"title": shelf.get("title", "For You"), "contents": contents})
    return shelves


def _extract_chart_section(section: Any) -> list:
    """Flatten a chart section (list or {items/results: [...]}) into a plain list."""
    if not section:
        return []
    items = section if isinstance(section, list) else (
        section.get("items") or section.get("results") or []
    )
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        thumbs = best_thumbnails_list(item.get("thumbnails") or [])
        item["thumbnails"] = thumbs
        item["thumbnail"]  = thumbs[0]["url"] if thumbs else ""
        out.append(item)
    return out


def _normalise_charts(raw: dict, country: str) -> dict:
    return {
        "country":  country,
        "songs":    [norm_track(t) for t in _extract_chart_section(raw.get("songs"))],
        "videos":   [norm_track(t) for t in _extract_chart_section(raw.get("videos"))
                     if t.get("videoId")],
        "artists":  [norm_artist_result(a) for a in _extract_chart_section(raw.get("artists"))],
        "trending": [norm_track(t) for t in _extract_chart_section(raw.get("trending"))
                     if t.get("videoId")],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Cache-Control header helper
# ─────────────────────────────────────────────────────────────────────────────

def _cc(seconds: int) -> dict:
    """Return a dict suitable for response.headers.update()."""
    return {"Cache-Control": f"public, max-age={seconds}, stale-while-revalidate=60"}


# ─────────────────────────────────────────────────────────────────────────────
#  Startup — non-blocking so Render health check passes immediately
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """
    Schedule background warm-up as a task.
    The server starts accepting requests BEFORE warm-up completes, which
    is exactly what Render needs (it checks /health within 30 s of start).
    """
    asyncio.create_task(_background_warmup())


async def _background_warmup():
    await asyncio.sleep(3)   # give uvicorn time to fully start

    # Initialise the default locale instance
    try:
        await run(get_ytm)
    except Exception:
        pass

    # Pre-fill the two most-requested caches so first users hit the cache
    for coro in (_warm_home("ZZ", "en"), _warm_charts("ZZ", "en")):
        try:
            await coro
        except Exception:
            pass


async def _warm_home(country: str, language: str) -> None:
    raw     = await run(get_ytm(country, language).get_home, 6)
    shelves = _normalise_home(raw or [])
    _cache_medium.set(f"home:{country}:{language}:6", shelves)


async def _warm_charts(country: str, language: str) -> None:
    raw    = await run(get_ytm(country, language).get_charts, country)
    result = _normalise_charts(raw, country)
    _cache_medium.set(f"charts:{country}:{language}", result)


# ─────────────────────────────────────────────────────────────────────────────
#  Health + admin
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    return {
        "status":         "ok",
        "version":        "5.0.0",
        "uptime_seconds": round(time.monotonic() - _START_TIME),
        "ytm_instances":  len(_ytm_pool),
    }


@app.get("/cache_stats", tags=["meta"])
async def cache_stats():
    """Live cache entry counts — handy for Render dashboard monitoring."""
    return {
        "short":    len(_cache_short),
        "medium":   len(_cache_medium),
        "long":     len(_cache_long),
        "upnext":   len(_upnext_store),
        "ytm_pool": len(_ytm_pool),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Search
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/search")
async def search(
    response: Response,
    query:   str,
    filter:  Optional[str] = Query(
        None, description="songs | videos | albums | artists | playlists | podcasts"
    ),
    scope:   Optional[str] = Query(None),
    limit:   int  = Query(20, ge=1, le=50),
    offset:  int  = Query(0,  ge=0, description="Pagination offset"),
    ignore_spelling: bool = False,
    country:  str = Query("ZZ", description="ISO 3166-1 alpha-2, e.g. IN US GB"),
    language: str = Query("en", description="BCP-47, e.g. en hi fr"),
):
    """
    Search YouTube Music.
    Returns { results, count, total, hasMore, offset, limit, country }.
    All items normalised with max-quality thumbnail at top level.
    """
    fetch_limit = min(offset + limit, 50)
    cache_key   = f"search:{query}:{filter}:{fetch_limit}:{country}:{language}"
    cached      = _cache_short.get(cache_key)

    if cached is None:
        try:
            raw    = await run(
                get_ytm(country, language).search,
                query, filter, scope, fetch_limit, ignore_spelling,
            )
            cached = norm_search_results(raw or [], filter)
            _cache_short.set(cache_key, cached)
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    page = cached[offset: offset + limit]
    response.headers.update(_cc(60))
    return {
        "results": page,
        "count":   len(page),
        "total":   len(cached),
        "hasMore": len(cached) > offset + limit,
        "offset":  offset,
        "limit":   limit,
        "country": country,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Search suggestions
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/search_suggestions")
async def search_suggestions(
    response: Response,
    query:    str,
    detailed_runs: bool = False,
    language: str = Query("en"),
):
    cache_key = f"suggest:{query}:{language}"
    cached    = _cache_short.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(30))
        return cached
    try:
        raw = await run(
            get_ytm("ZZ", language).get_search_suggestions, query, detailed_runs
        )
        suggestions: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    suggestions.append(item)
                elif isinstance(item, dict):
                    text = item.get("query") or item.get("text") or item.get("suggestion", "")
                    if text:
                        suggestions.append(text)
        _cache_short.set(cache_key, suggestions, ttl=60)
        response.headers.update(_cc(30))
        return suggestions
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Artist
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/artist/{artist_id}")
async def get_artist(
    response:  Response,
    artist_id: str,
    language:  str = Query("en"),
):
    cache_key = f"artist:{artist_id}:{language}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(600))
        return cached
    try:
        data = await run(get_ytm("ZZ", language).get_artist, artist_id)
        if not data:
            raise HTTPException(404, "Artist not found")
        data["thumbnails"] = best_thumbnails_list(data.get("thumbnails") or [])
        data["thumbnail"]  = data["thumbnails"][0]["url"] if data["thumbnails"] else ""
        for section in ("songs", "albums", "singles", "videos", "related"):
            sd = data.get(section, {})
            if not isinstance(sd, dict):
                continue
            normed = []
            for item in (sd.get("results") or sd.get("items") or []):
                if not isinstance(item, dict):
                    continue
                item["thumbnails"] = best_thumbnails_list(item.get("thumbnails") or [])
                item["thumbnail"]  = item["thumbnails"][0]["url"] if item["thumbnails"] else ""
                normed.append(item)
            sd["results"] = normed
            data[section] = sd
        _cache_long.set(cache_key, data)
        response.headers.update(_cc(600))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Artist — all songs
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/artist/{artist_id}/songs")
async def get_artist_songs(
    response:  Response,
    artist_id: str,
    limit:     int = Query(100, ge=1, le=200),
    language:  str = Query("en"),
):
    cache_key = f"artist_songs:{artist_id}:{limit}:{language}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(600))
        return cached
    try:
        ytm         = get_ytm("ZZ", language)
        artist_data = await run(ytm.get_artist, artist_id)
        if not artist_data:
            raise HTTPException(404, "Artist not found")
        artist_name   = artist_data.get("name", "")
        songs_section = artist_data.get("songs", {})
        params_token  = None
        if isinstance(songs_section, dict):
            params_token = songs_section.get("params") or songs_section.get("browseId")

        songs_raw: list = []

        # Strategy 1: follow the "all songs" browse token
        if params_token:
            try:
                more = await run(ytm.get_artist_albums, artist_id, params_token)
                if isinstance(more, list):
                    songs_raw = more
                elif isinstance(more, dict):
                    songs_raw = (
                        more.get("results") or more.get("items") or
                        more.get("songs") or more.get("tracks") or []
                    )
            except Exception:
                songs_raw = []

        # Strategy 2: initial songs already on the artist page
        if not songs_raw and isinstance(songs_section, dict):
            songs_raw = list(songs_section.get("results") or songs_section.get("items") or [])

        # Strategy 3: search fallback
        if not songs_raw and artist_name:
            try:
                sr = await run(ytm.search, artist_name, "songs", None, min(limit, 50), False)
                songs_raw = sr if isinstance(sr, list) else []
            except Exception:
                pass

        tracks = [norm_track(t) for t in songs_raw if isinstance(t, dict) and t.get("videoId")]
        seen, deduped = set(), []
        for t in tracks:
            vid = t.get("videoId", "")
            if vid and vid not in seen:
                seen.add(vid); deduped.append(t)

        result = {
            "artistId": artist_id,
            "name":     artist_name,
            "songs":    deduped[:limit],
            "total":    len(deduped),
        }
        _cache_long.set(cache_key, result)
        response.headers.update(_cc(600))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Album
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/album/{album_id}")
async def get_album(
    response: Response,
    album_id: str,
    language: str = Query("en"),
):
    cache_key = f"album:{album_id}:{language}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(3600))
        return cached
    try:
        data = await run(get_ytm("ZZ", language).get_album, album_id)
        if not data:
            raise HTTPException(404, "Album not found")
        data["thumbnails"] = best_thumbnails_list(data.get("thumbnails") or [])
        data["thumbnail"]  = data["thumbnails"][0]["url"] if data["thumbnails"] else ""
        data["tracks"]     = [norm_track(t) for t in (data.get("tracks") or [])]
        _cache_long.set(cache_key, data)
        response.headers.update(_cc(3600))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Song metadata
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/song/{video_id}")
async def get_song(response: Response, video_id: str):
    cache_key = f"song:{video_id}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(3600))
        return cached
    try:
        data = await run(get_ytm().get_song, video_id)
        raw  = (
            data.get("thumbnail", {}).get("thumbnails") or
            data.get("thumbnails") or []
        )
        thumbs = best_thumbnails_list(raw)
        data["thumbnails"] = thumbs
        data["thumbnail"]  = thumbs[0]["url"] if thumbs else ""
        _cache_long.set(cache_key, data)
        response.headers.update(_cc(3600))
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Stream metadata (frontend YouTube IFrame + download)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/stream/{video_id}")
async def get_stream(response: Response, video_id: str):
    """
    Returns metadata + max-quality thumbnails.
    url / audio_url / stream_url are all the same YouTube watch URL;
    the frontend tries all three key names for compatibility.
    """
    cache_key = f"stream:{video_id}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(3600))
        return cached
    try:
        song_data = await run(get_ytm().get_song, video_id)
        vd = song_data.get("videoDetails", {})
        if not vd:
            raise HTTPException(404, "Video not found")
        raw = (
            vd.get("thumbnail", {}).get("thumbnails") or
            song_data.get("thumbnail", {}).get("thumbnails") or []
        )
        thumbs    = best_thumbnails_list(raw)
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        result = {
            "video_id":          video_id,
            "videoId":           video_id,
            "url":               watch_url,
            "audio_url":         watch_url,
            "stream_url":        watch_url,
            "title":             vd.get("title", ""),
            "artist":            vd.get("author", ""),
            "channel_id":        vd.get("channelId", ""),
            "duration_seconds":  int(vd.get("lengthSeconds") or 0),
            "views":             int(vd.get("viewCount") or 0),
            "keywords":          vd.get("keywords", [])[:10],
            "is_live":           vd.get("isLiveContent", False),
            "thumbnails":        thumbs,
            "thumbnail":         thumbs[0]["url"] if thumbs else "",
        }
        _cache_long.set(cache_key, result)
        response.headers.update(_cc(3600))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Now Playing  (stream metadata + related in one parallel call)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/now_playing/{video_id}")
async def now_playing(
    response:      Response,
    video_id:      str,
    related_limit: int = Query(10, ge=1, le=30),
    country:       str = Query("ZZ"),
    language:      str = Query("en"),
):
    """
    One-shot endpoint: stream metadata + related tracks in parallel.
    Returns { videoId, stream, related }.
    """
    cache_key = f"now_playing:{video_id}:{related_limit}:{country}:{language}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(1800))
        return cached

    ytm = get_ytm(country, language)
    song_data, watch_data = await asyncio.gather(
        run(ytm.get_song, video_id),
        run(ytm.get_watch_playlist, video_id, None, related_limit + 1),
        return_exceptions=True,
    )

    stream: dict = {}
    if isinstance(song_data, dict):
        vd  = song_data.get("videoDetails", {})
        raw = (
            vd.get("thumbnail", {}).get("thumbnails") or
            song_data.get("thumbnail", {}).get("thumbnails") or []
        )
        thumbs    = best_thumbnails_list(raw)
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        stream = {
            "videoId":          video_id,
            "url":              watch_url,
            "audio_url":        watch_url,
            "title":            vd.get("title", ""),
            "artist":           vd.get("author", ""),
            "duration_seconds": int(vd.get("lengthSeconds") or 0),
            "thumbnails":       thumbs,
            "thumbnail":        thumbs[0]["url"] if thumbs else "",
        }

    related: list = []
    if isinstance(watch_data, dict):
        tracks_raw = watch_data.get("tracks") or []
        if tracks_raw and tracks_raw[0].get("videoId") == video_id:
            tracks_raw = tracks_raw[1:]
        related = [norm_track(t) for t in tracks_raw[:related_limit] if t.get("videoId")]

    result = {"videoId": video_id, "stream": stream, "related": related}
    _cache_long.set(cache_key, result, ttl=1800)
    response.headers.update(_cc(1800))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Related songs
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/related_songs/{video_id}")
async def get_related_songs(
    response:  Response,
    video_id:  str,
    limit:     int = Query(15, ge=1, le=50),
    country:   str = Query("ZZ"),
    language:  str = Query("en"),
):
    cache_key = f"related:{video_id}:{limit}:{country}:{language}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(1800))
        return cached
    try:
        raw        = await run(get_ytm(country, language).get_watch_playlist, video_id, None, limit + 1)
        tracks_raw = raw.get("tracks") or []
        if tracks_raw and tracks_raw[0].get("videoId") == video_id:
            tracks_raw = tracks_raw[1:]
        tracks = [norm_track(t) for t in tracks_raw[:limit] if t.get("videoId")]
        result = {"videoId": video_id, "tracks": tracks, "count": len(tracks)}
        _cache_long.set(cache_key, result, ttl=1800)
        response.headers.update(_cc(1800))
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Up-Next  (watch queue / radio)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/upnext/{video_id}")
async def get_upnext(
    response:      Response,
    video_id:      str,
    limit:         int  = Query(20, ge=5, le=50),
    force_refresh: bool = Query(False),
    country:       str = Query("ZZ"),
    language:      str = Query("en"),
):
    """
    Up-Next queue for a video.  Cached in-process for 2 h.
    Pass force_refresh=true when the user manually picks a new song.
    """
    store_key = f"{video_id}:{country}:{language}"
    now       = time.time()

    with _upnext_lock:
        existing = _upnext_store.get(store_key)

    if existing and not force_refresh:
        if now - existing.get("created_at", 0) < _UPNEXT_TTL:
            response.headers.update(_cc(300))
            return existing

    try:
        raw        = await run(get_ytm(country, language).get_watch_playlist, video_id, None, limit)
        tracks_raw = raw.get("tracks") or []
        if tracks_raw and tracks_raw[0].get("videoId") == video_id:
            tracks_raw = tracks_raw[1:]
        tracks = [norm_track(t) for t in tracks_raw if t.get("videoId")]
        queue  = {
            "origin_video_id": video_id,
            "tracks":          tracks,
            "count":           len(tracks),
            "created_at":      now,
            "country":         country,
        }
        # Proper LRU eviction using OrderedDict
        with _upnext_lock:
            if store_key in _upnext_store:
                _upnext_store.move_to_end(store_key)
            _upnext_store[store_key] = queue
            while len(_upnext_store) > _UPNEXT_MAX:
                _upnext_store.popitem(last=False)

        response.headers.update(_cc(300))
        return queue
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.delete("/upnext/{video_id}")
async def reset_upnext(
    video_id: str,
    country:  str = Query("ZZ"),
    language: str = Query("en"),
):
    store_key = f"{video_id}:{country}:{language}"
    with _upnext_lock:
        _upnext_store.pop(store_key, None)
    return {"cleared": store_key}


# ─────────────────────────────────────────────────────────────────────────────
#  Playlist
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/playlist/{playlist_id}")
async def get_playlist(
    response:     Response,
    playlist_id:  str,
    limit:        int  = Query(100, ge=1, le=500),
    related:      bool = False,
    suggestions_limit: int = 0,
    language:     str = Query("en"),
):
    """Full playlist — normalised tracks, max-quality thumbnails. Handles VL-prefix IDs."""
    clean_id  = playlist_id[2:] if playlist_id.startswith("VL") else playlist_id
    cache_key = f"playlist:{clean_id}:{limit}:{language}"
    cached    = _cache_medium.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(300))
        return cached
    try:
        data = await run(
            get_ytm("ZZ", language).get_playlist,
            clean_id, limit, related, suggestions_limit,
        )
        if not data:
            raise HTTPException(404, "Playlist not found")
        data["tracks"]     = [norm_track(t) for t in (data.get("tracks") or []) if isinstance(t, dict)]
        data["thumbnails"] = best_thumbnails_list(data.get("thumbnails") or [])
        data["thumbnail"]  = data["thumbnails"][0]["url"] if data["thumbnails"] else ""
        _cache_medium.set(cache_key, data)
        response.headers.update(_cc(300))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Podcast
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/podcast/{podcast_id}")
async def get_podcast(
    response:   Response,
    podcast_id: str,
    limit:      int = Query(50, ge=1, le=200),
    language:   str = Query("en"),
):
    """
    Full podcast page with normalised episodes.
    Returns { podcastId, title, author, description, thumbnail, thumbnails, episodes, total }.
    Falls back from get_podcast → get_playlist and from clean_id → original_id.
    """
    # Strip "VL" prefix only when it is a plain VL prefix, not VLM… (podcast playlists)
    clean_id  = podcast_id[2:] if (podcast_id.startswith("VL") and not podcast_id.startswith("VLM")) else podcast_id
    cache_key = f"podcast:{clean_id}:{limit}:{language}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(1800))
        return cached

    ytm          = get_ytm("ZZ", language)
    episodes_raw: list = []
    meta: dict         = {}

    # Four attempts — stop as soon as we have episodes
    for fn, *fn_args in [
        (ytm.get_podcast, clean_id),
        (ytm.get_playlist, clean_id, limit),
        (ytm.get_podcast, podcast_id),
        (ytm.get_playlist, podcast_id, limit),
    ]:
        if episodes_raw:
            break
        try:
            data = await run(fn, *fn_args)
            if isinstance(data, dict) and data:
                if not meta:
                    meta = data
                episodes_raw = data.get("episodes") or data.get("tracks") or []
        except Exception:
            pass

    if not meta and not episodes_raw:
        raise HTTPException(404, "Podcast not found")

    def _norm_episode(ep: dict) -> dict:
        raw_t = ep.get("thumbnails") or ep.get("thumbnail") or []
        if isinstance(raw_t, str):
            raw_t = [{"url": raw_t, "width": 0}]
        thumbs = best_thumbnails_list(raw_t)
        dur    = ep.get("duration") or ep.get("durationSeconds") or ""
        if isinstance(dur, int) and dur > 0:
            m, s = divmod(dur, 60)
            h, m = divmod(m, 60)
            dur  = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        artists = ep.get("artists") or ep.get("author") or ""
        if isinstance(artists, list):
            artists = ", ".join(
                a.get("name", "") if isinstance(a, dict) else str(a) for a in artists
            )
        return {
            "videoId":     ep.get("videoId", ""),
            "title":       ep.get("title", ""),
            "artist":      artists or "",
            "thumbnail":   thumbs[0]["url"] if thumbs else "",
            "thumbnails":  thumbs,
            "duration":    str(dur),
            "date":        ep.get("date") or ep.get("publishedTime") or "",
            "description": ep.get("description") or ep.get("shortDescription") or "",
        }

    episodes    = [_norm_episode(ep) for ep in episodes_raw
                   if isinstance(ep, dict) and ep.get("videoId")][:limit]
    meta_thumbs = best_thumbnails_list(meta.get("thumbnails") or [])
    author      = meta.get("author") or ""
    if isinstance(author, dict):
        author = author.get("name", "")
    if not author and isinstance(meta.get("channel"), dict):
        author = meta["channel"].get("name", "")

    result = {
        "podcastId":   clean_id,
        "title":       meta.get("title", ""),
        "author":      author,
        "description": meta.get("description") or meta.get("shortDescription") or "",
        "thumbnail":   meta_thumbs[0]["url"] if meta_thumbs else "",
        "thumbnails":  meta_thumbs,
        "episodes":    episodes,
        "total":       len(episodes),
    }
    _cache_long.set(cache_key, result)
    response.headers.update(_cc(1800))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Watch playlist  (raw — mostly for internal use)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/watch_playlist")
async def get_watch_playlist(
    response:    Response,
    video_id:    Optional[str] = None,
    playlist_id: Optional[str] = None,
    limit:       int           = 25,
    params:      Optional[str] = None,
    country:     str = Query("ZZ"),
    language:    str = Query("en"),
):
    try:
        data = await run(
            get_ytm(country, language).get_watch_playlist,
            video_id, playlist_id, limit, params,
        )
        for t in (data.get("tracks") or []):
            if isinstance(t, dict):
                t["thumbnails"] = best_thumbnails_list(t.get("thumbnails") or [])
                t["thumbnail"]  = t["thumbnails"][0]["url"] if t["thumbnails"] else ""
        response.headers.update(_cc(300))
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Lyrics  (by browseId)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/lyrics/{browse_id}")
async def get_lyrics(response: Response, browse_id: str):
    cache_key = f"lyrics:{browse_id}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(7200))
        return cached
    try:
        data = await run(get_ytm().get_lyrics, browse_id)
        _cache_long.set(cache_key, data)
        response.headers.update(_cc(7200))
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Lyrics by video ID  (no separate browseId lookup needed)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/lyrics_by_video/{video_id}")
async def get_lyrics_by_video(response: Response, video_id: str):
    """
    Get lyrics without a separate browseId lookup.
    Returns { lyricsId, lyrics, source } or { lyricsId: null, error: "…" }.
    """
    cache_key = f"lyrics_vid:{video_id}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(7200))
        return cached

    ytm = get_ytm()
    try:
        watch     = await run(ytm.get_watch_playlist, video_id, None, 1)
        lyrics_id = watch.get("lyrics")
    except Exception:
        lyrics_id = None

    if not lyrics_id:
        result = {"lyricsId": None, "lyrics": None, "error": "Lyrics not available"}
        _cache_long.set(cache_key, result, ttl=600)
        return result

    try:
        data   = await run(ytm.get_lyrics, lyrics_id)
        result = {
            "lyricsId": lyrics_id,
            "lyrics":   data.get("lyrics") if isinstance(data, dict) else data,
            "source":   data.get("source", "") if isinstance(data, dict) else "",
        }
        _cache_long.set(cache_key, result)
        response.headers.update(_cc(7200))
        return result
    except Exception as e:
        result = {"lyricsId": lyrics_id, "lyrics": None, "error": str(e)}
        _cache_long.set(cache_key, result, ttl=300)
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Home feed  (country-aware)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/home")
async def get_home(
    response: Response,
    limit:    int = Query(6, ge=1, le=15),
    country:  str = Query("ZZ"),
    language: str = Query("en"),
):
    """Home feed shelves, normalised. Pass ?country=IN for Indian content."""
    cache_key = f"home:{country}:{language}:{limit}"
    cached    = _cache_medium.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(300))
        return cached
    try:
        raw     = await run(get_ytm(country, language).get_home, limit)
        shelves = _normalise_home(raw or [])
        _cache_medium.set(cache_key, shelves)
        response.headers.update(_cc(300))
        return shelves
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Charts  (country-aware)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/charts")
async def get_charts(
    response: Response,
    country:  str = "ZZ",
    language: str = Query("en"),
):
    """Music charts.  Returns { country, songs, videos, artists, trending }."""
    cache_key = f"charts:{country}:{language}"
    cached    = _cache_medium.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(600))
        return cached
    try:
        raw    = await run(get_ytm(country, language).get_charts, country)
        result = _normalise_charts(raw, country)
        _cache_medium.set(cache_key, result)
        response.headers.update(_cc(600))
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Trending  — delegates to the shared charts cache key, never double-fetches
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/trending")
async def get_trending(
    response: Response,
    country:  str = Query("ZZ"),
    language: str = Query("en"),
    limit:    int = Query(20, ge=1, le=50),
):
    """Trending songs for a country.  Returns { country, trending, count }."""
    charts_key = f"charts:{country}:{language}"
    charts     = _cache_medium.get(charts_key)

    if charts is None:
        try:
            raw    = await run(get_ytm(country, language).get_charts, country)
            charts = _normalise_charts(raw, country)
            _cache_medium.set(charts_key, charts)
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    trending = charts.get("trending") or charts.get("songs") or []
    response.headers.update(_cc(600))
    return {"country": country, "trending": trending[:limit], "count": min(len(trending), limit)}


# ─────────────────────────────────────────────────────────────────────────────
#  Top Playlists  (country-aware)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/top_playlists")
async def get_top_playlists(
    response: Response,
    country:  str = Query("ZZ"),
    language: str = Query("en"),
    limit:    int = Query(16, ge=1, le=50),
):
    cache_key  = f"top_playlists:{country}:{language}:{limit}"
    cached     = _cache_medium.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(600))
        return cached

    ytm       = get_ytm(country, language)
    playlists: list[dict] = []

    # Source 1: country charts
    try:
        charts_raw = await run(ytm.get_charts, country)
        for sk in ("trending", "songs"):
            section = charts_raw.get(sk, {})
            items   = section if isinstance(section, list) else (
                section.get("items") or section.get("results") or []
            )
            for item in items:
                if not isinstance(item, dict):
                    continue
                bid = item.get("browseId") or item.get("playlistId")
                if bid and item.get("title"):
                    thumbs = best_thumbnails_list(item.get("thumbnails") or [])
                    playlists.append({
                        "browseId":   bid,
                        "title":      item.get("title", ""),
                        "subtitle":   item.get("subtitle", "") or item.get("author", ""),
                        "thumbnails": thumbs,
                        "thumbnail":  thumbs[0]["url"] if thumbs else "",
                    })
    except Exception:
        pass

    # Source 2: mood categories (fills out the list if charts gave few results)
    if len(playlists) < limit:
        try:
            cats         = await run(ytm.get_mood_categories)
            first_params = None
            if isinstance(cats, dict):
                for items in cats.values():
                    if isinstance(items, list) and items:
                        p = items[0].get("params")
                        if p:
                            first_params = p
                            break
            elif isinstance(cats, list) and cats:
                first_params = cats[0].get("params")

            if first_params:
                mood_raw = await run(ytm.get_mood_playlists, first_params)
                for section in (mood_raw or []):
                    if not isinstance(section, dict):
                        continue
                    for item in (section.get("contents") or section.get("playlists") or []):
                        if not isinstance(item, dict):
                            continue
                        bid = item.get("playlistId") or item.get("browseId")
                        if bid:
                            thumbs = best_thumbnails_list(item.get("thumbnails") or [])
                            playlists.append({
                                "browseId":   bid,
                                "title":      item.get("title", ""),
                                "subtitle":   item.get("subtitle", ""),
                                "thumbnails": thumbs,
                                "thumbnail":  thumbs[0]["url"] if thumbs else "",
                            })
        except Exception:
            pass

    # Deduplicate by browseId
    seen, deduped = set(), []
    for p in playlists:
        bid = p.get("browseId", "")
        if bid and bid not in seen:
            seen.add(bid); deduped.append(p)

    result = deduped[:limit]
    _cache_medium.set(cache_key, result)
    response.headers.update(_cc(600))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Mood categories  (country-aware)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/mood_categories")
async def get_mood_categories(
    response: Response,
    country:  str = Query("ZZ"),
    language: str = Query("en"),
):
    cache_key = f"mood_categories:{country}:{language}"
    cached    = _cache_medium.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(600))
        return cached
    try:
        raw        = await run(get_ytm(country, language).get_mood_categories)
        categories: list[dict] = []
        if isinstance(raw, dict):
            for section_title, items in raw.items():
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        thumbs = best_thumbnails_list(item.get("thumbnails") or [])
                        categories.append({
                            "title":      item.get("title", ""),
                            "params":     item.get("params", ""),
                            "section":    section_title,
                            "thumbnails": thumbs,
                            "thumbnail":  thumbs[0]["url"] if thumbs else "",
                        })
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                thumbs = best_thumbnails_list(item.get("thumbnails") or [])
                categories.append({
                    "title":      item.get("title", ""),
                    "params":     item.get("params", ""),
                    "thumbnails": thumbs,
                    "thumbnail":  thumbs[0]["url"] if thumbs else "",
                })
        _cache_medium.set(cache_key, categories)
        response.headers.update(_cc(600))
        return categories
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Mood playlists  (country-aware)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/mood_playlists/{params}")
async def get_mood_playlists(
    response: Response,
    params:   str,
    country:  str = Query("ZZ"),
    language: str = Query("en"),
):
    cache_key = f"mood_playlists:{params}:{country}:{language}"
    cached    = _cache_medium.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(600))
        return cached
    try:
        raw       = await run(get_ytm(country, language).get_mood_playlists, params)
        playlists: list[dict] = []
        for section in (raw if isinstance(raw, list) else []):
            if not isinstance(section, dict):
                continue
            contents = section.get("contents") or section.get("playlists") or []
            if isinstance(contents, list):
                for item in contents:
                    if not isinstance(item, dict):
                        continue
                    thumbs = best_thumbnails_list(item.get("thumbnails") or [])
                    playlists.append({
                        "browseId":   item.get("playlistId") or item.get("browseId", ""),
                        "title":      item.get("title", ""),
                        "subtitle":   item.get("subtitle", ""),
                        "thumbnails": thumbs,
                        "thumbnail":  thumbs[0]["url"] if thumbs else "",
                    })
            else:
                thumbs = best_thumbnails_list(section.get("thumbnails") or [])
                playlists.append({
                    "browseId":   section.get("playlistId") or section.get("browseId", ""),
                    "title":      section.get("title", ""),
                    "subtitle":   section.get("subtitle", ""),
                    "thumbnails": thumbs,
                    "thumbnail":  thumbs[0]["url"] if thumbs else "",
                })
        _cache_medium.set(cache_key, playlists)
        response.headers.update(_cc(600))
        return playlists
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Explore  (country-aware)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/explore")
async def get_explore(
    response: Response,
    country:  str = Query("ZZ"),
    language: str = Query("en"),
):
    cache_key = f"explore:{country}:{language}"
    cached    = _cache_medium.get(cache_key)
    if cached is not None:
        response.headers.update(_cc(600))
        return cached
    try:
        data = await run(get_ytm(country, language).get_explore)
        _cache_medium.set(cache_key, data)
        response.headers.update(_cc(600))
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  User endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/user/{channel_id}")
async def get_user(
    response:   Response,
    channel_id: str,
    language:   str = Query("en"),
):
    cache_key = f"user:{channel_id}:{language}"
    cached    = _cache_long.get(cache_key)
    if cached is not None:
        return cached
    try:
        data = await run(get_ytm("ZZ", language).get_user, channel_id)
        _cache_long.set(cache_key, data)
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/user_playlists/{channel_id}")
async def get_user_playlists(
    response:   Response,
    channel_id: str,
    params:     Optional[str] = None,
    language:   str = Query("en"),
):
    try:
        data = await run(get_ytm("ZZ", language).get_user_playlists, channel_id, params)
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))
