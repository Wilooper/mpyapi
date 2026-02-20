"""
YouTube Music API Wrapper — Optimized for Render 512 MB free tier
=================================================================
Improvements over v1:
 - Highest-quality thumbnails everywhere (helper picks largest by width)
 - In-memory TTL cache (no Redis needed) to avoid hammering YT and save RAM
 - Async-friendly: blocking ytmusicapi calls run in a thread pool so the
   event loop is never blocked
 - Single shared YTMusic instance (thread-safe for reads)
 - Proper CORS, GZip compression, and health-check endpoint
 - /upnext/{video_id} — returns a watch queue for the given song and
   remembers the queue in memory so playing the "next" track does NOT
   trigger a new fetch; only a manual song selection resets the queue
 - Fixed /charts — normalises the raw ytmusicapi dict into consistent shapes
 - Fixed /mood_categories + /mood_playlists — normalised response
 - /stream returns highest-quality thumbnail + all useful metadata
 - Startup warm-up so the first real request is fast
"""

from __future__ import annotations

import asyncio
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from ytmusicapi import YTMusic

# ─────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="YouTube Music API Wrapper",
    description="Optimized FastAPI wrapper for unauthenticated ytmusicapi.",
    version="2.0.0",
)

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Thread pool — keep small on 512 MB RAM (ytmusicapi is I/O-bound)
_executor = ThreadPoolExecutor(max_workers=4)

# Single YTMusic instance (unauthenticated, thread-safe for reads)
_ytm: YTMusic | None = None
_ytm_lock = threading.Lock()


def get_ytm() -> YTMusic:
    global _ytm
    if _ytm is None:
        with _ytm_lock:
            if _ytm is None:
                _ytm = YTMusic()
    return _ytm


async def run(fn, *args, **kwargs):
    """Run a blocking call in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


# ─────────────────────────────────────────────────────────────
#  TTL Cache  (simple LRU with expiry — no external deps)
# ─────────────────────────────────────────────────────────────

class TTLCache:
    """Thread-safe in-memory cache with per-entry TTL and max size."""

    def __init__(self, maxsize: int = 256, ttl: int = 300):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self._store:
                return None
            value, expires = self._store[key]
            if time.monotonic() > expires:
                del self._store[key]
                return None
            # Move to end (LRU)
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


# Different TTLs for different data types
_cache_short   = TTLCache(maxsize=128, ttl=120)   # 2 min  — search, suggestions
_cache_medium  = TTLCache(maxsize=128, ttl=600)   # 10 min — home, charts, mood
_cache_long    = TTLCache(maxsize=256, ttl=3600)  # 1 hr   — artist, album, song


# ─────────────────────────────────────────────────────────────
#  Up-Next queue store
#  Key: video_id that STARTED the queue
#  Value: { "tracks": [...], "created_at": float }
#  The queue is NOT regenerated while user is consuming it.
#  Only manual play (POST /upnext/reset) or a brand-new song
#  that has no queue yet creates a new one.
# ─────────────────────────────────────────────────────────────

_upnext_store: Dict[str, Dict] = {}   # { origin_video_id -> {tracks, created_at} }
_UPNEXT_TTL = 7200                    # 2 hours — after that we'll refresh


# ─────────────────────────────────────────────────────────────
#  Thumbnail helpers
# ─────────────────────────────────────────────────────────────

def best_thumbnail(thumbnails: list | None) -> str:
    """Return the URL of the highest-resolution thumbnail."""
    if not thumbnails:
        return ""
    # Sort descending by width (or height as fallback)
    try:
        sorted_thumbs = sorted(
            thumbnails,
            key=lambda t: (t.get("width", 0) or 0, t.get("height", 0) or 0),
            reverse=True,
        )
        return sorted_thumbs[0].get("url", "") if sorted_thumbs else ""
    except Exception:
        return thumbnails[-1].get("url", "") if thumbnails else ""


def best_thumbnails_list(thumbnails: list | None) -> list:
    """Return thumbnails sorted best-first, deduplicated by URL."""
    if not thumbnails:
        return []
    try:
        seen, out = set(), []
        for t in sorted(
            thumbnails,
            key=lambda t: (t.get("width", 0) or 0),
            reverse=True,
        ):
            url = t.get("url", "")
            if url and url not in seen:
                seen.add(url)
                out.append(t)
        return out
    except Exception:
        return thumbnails


# ─────────────────────────────────────────────────────────────
#  Normalisation helpers  (make every entity shape consistent)
# ─────────────────────────────────────────────────────────────

def _norm_artists(artists: Any) -> list:
    if not artists:
        return []
    if isinstance(artists, str):
        return [{"name": artists}]
    if isinstance(artists, list):
        out = []
        for a in artists:
            if isinstance(a, str):
                out.append({"name": a})
            elif isinstance(a, dict):
                out.append({"name": a.get("name", ""), "id": a.get("id", "")})
        return out
    return []


def norm_track(t: dict) -> dict:
    """Normalise a track/song dict to a consistent shape."""
    thumbs = best_thumbnails_list(
        t.get("thumbnails") or
        t.get("thumbnail") or []
    )
    return {
        "videoId":    t.get("videoId") or t.get("videoId", ""),
        "title":      t.get("title", ""),
        "artists":    _norm_artists(t.get("artists") or t.get("artist")),
        "album":      (t.get("album") or {}).get("name", "") if isinstance(t.get("album"), dict) else (t.get("album") or ""),
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
        "browseId":  p.get("browseId", "") or p.get("playlistId", ""),
        "title":     p.get("title", ""),
        "author":    p.get("author", ""),
        "itemCount": p.get("itemCount", ""),
        "thumbnails": thumbs,
        "thumbnail":  thumbs[0]["url"] if thumbs else "",
    }


def norm_podcast_result(p: dict) -> dict:
    thumbs = best_thumbnails_list(p.get("thumbnails") or [])
    return {
        "browseId":   p.get("browseId", "") or p.get("podcastId", ""),
        "title":      p.get("title", ""),
        "author":     p.get("author", "") or (
            ", ".join(a.get("name", "") for a in _norm_artists(p.get("artists")))
        ),
        "thumbnails": thumbs,
        "thumbnail":  thumbs[0]["url"] if thumbs else "",
    }


def norm_search_results(raw: list, filter_type: str | None) -> list:
    """Normalise a list of raw search results based on filter type."""
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rt = (item.get("resultType") or filter_type or "").lower()
        if rt in ("song", "songs", "video", "videos"):
            n = norm_track(item)
            n["resultType"] = "song" if "video" not in rt else "video"
            out.append(n)
        elif rt in ("artist", "artists"):
            n = norm_artist_result(item)
            n["resultType"] = "artist"
            out.append(n)
        elif rt in ("album", "albums", "single", "singles", "ep"):
            n = norm_album_result(item)
            n["resultType"] = "album"
            out.append(n)
        elif rt in ("playlist", "playlists"):
            n = norm_playlist_result(item)
            n["resultType"] = "playlist"
            out.append(n)
        elif rt in ("podcast", "podcasts", "episode", "episodes"):
            n = norm_podcast_result(item)
            n["resultType"] = "podcast"
            out.append(n)
        else:
            # Unknown — pass through with thumbnail fix
            thumbs = best_thumbnails_list(item.get("thumbnails") or [])
            item["thumbnails"] = thumbs
            item["thumbnail"] = thumbs[0]["url"] if thumbs else ""
            out.append(item)
    return out


# ─────────────────────────────────────────────────────────────
#  Startup warm-up
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Pre-initialise the YTMusic client so first request is fast."""
    try:
        await run(get_ytm)
    except Exception:
        pass  # Non-fatal — will initialise on first request


# ─────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ─────────────────────────────────────────────────────────────
#  Search
# ─────────────────────────────────────────────────────────────

@app.get("/search")
async def search(
    query: str,
    filter: Optional[str] = Query(None, description="songs | videos | albums | artists | playlists | podcasts"),
    scope: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    ignore_spelling: bool = False,
):
    """Search YouTube Music with normalised, consistently shaped results."""
    cache_key = f"search:{query}:{filter}:{limit}"
    cached = _cache_short.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = await run(get_ytm().search, query, filter, scope, limit, ignore_spelling)
        result = norm_search_results(raw or [], filter)
        _cache_short.set(cache_key, result)
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Search suggestions
# ─────────────────────────────────────────────────────────────

@app.get("/search_suggestions")
async def search_suggestions(query: str, detailed_runs: bool = False):
    """Get search suggestions as a plain list of strings."""
    cache_key = f"suggest:{query}"
    cached = _cache_short.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = await run(get_ytm().get_search_suggestions, query, detailed_runs)
        # ytmusicapi can return list[str] or list[dict]
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
        return suggestions
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Artist
# ─────────────────────────────────────────────────────────────

@app.get("/artist/{artist_id}")
async def get_artist(
    artist_id: str,
    filter: Optional[str] = None,
    limit: int = 20,
    offset: Optional[int] = None,
):
    """Full artist page — thumbnails sorted best-first."""
    cache_key = f"artist:{artist_id}:{limit}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached

    try:
        data = await run(get_ytm().get_artist, artist_id)
        if not data:
            raise HTTPException(404, "Artist not found")

        # Fix thumbnails at top level
        data["thumbnails"] = best_thumbnails_list(data.get("thumbnails") or [])
        data["thumbnail"]  = data["thumbnails"][0]["url"] if data["thumbnails"] else ""

        # Fix thumbnails inside nested sections
        for section in ("songs", "albums", "singles", "videos", "related"):
            section_data = data.get(section, {})
            if not isinstance(section_data, dict):
                continue
            results = section_data.get("results") or section_data.get("items") or []
            for item in results:
                if isinstance(item, dict):
                    item["thumbnails"] = best_thumbnails_list(item.get("thumbnails") or [])
                    item["thumbnail"]  = item["thumbnails"][0]["url"] if item["thumbnails"] else ""

        _cache_long.set(cache_key, data)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Album
# ─────────────────────────────────────────────────────────────

@app.get("/album/{album_id}")
async def get_album(album_id: str):
    """Full album — tracks normalised, thumbnails best-first."""
    cache_key = f"album:{album_id}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached

    try:
        data = await run(get_ytm().get_album, album_id)
        if not data:
            raise HTTPException(404, "Album not found")

        data["thumbnails"] = best_thumbnails_list(data.get("thumbnails") or [])
        data["thumbnail"]  = data["thumbnails"][0]["url"] if data["thumbnails"] else ""

        tracks = data.get("tracks") or []
        data["tracks"] = [norm_track(t) for t in tracks]

        _cache_long.set(cache_key, data)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Song
# ─────────────────────────────────────────────────────────────

@app.get("/song/{video_id}")
async def get_song(video_id: str):
    """Song metadata with highest-quality thumbnail first."""
    cache_key = f"song:{video_id}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached

    try:
        data = await run(get_ytm().get_song, video_id)
        # Fix thumbnail inside the nested structure
        raw_thumbs = (
            data.get("thumbnail", {}).get("thumbnails") or
            data.get("thumbnails") or []
        )
        sorted_thumbs = best_thumbnails_list(raw_thumbs)
        data["thumbnails"] = sorted_thumbs
        data["thumbnail"]  = sorted_thumbs[0]["url"] if sorted_thumbs else ""

        _cache_long.set(cache_key, data)
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Stream metadata  (used by frontend to get videoId + metadata)
# ─────────────────────────────────────────────────────────────

@app.get("/stream/{video_id}")
async def get_stream(video_id: str):
    """
    Returns metadata + highest-quality thumbnails for a video.
    The frontend uses this to feed the YouTube IFrame Player.
    """
    cache_key = f"stream:{video_id}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached

    try:
        song_data = await run(get_ytm().get_song, video_id)
        vd = song_data.get("videoDetails", {})
        if not vd:
            raise HTTPException(404, "Video not found")

        raw_thumbs = (
            vd.get("thumbnail", {}).get("thumbnails") or
            song_data.get("thumbnail", {}).get("thumbnails") or []
        )
        sorted_thumbs = best_thumbnails_list(raw_thumbs)

        result = {
            "video_id":  video_id,
            "videoId":   video_id,
            "title":     vd.get("title", ""),
            "artist":    vd.get("author", ""),
            "channel_id": vd.get("channelId", ""),
            "duration_seconds": int(vd.get("lengthSeconds") or 0),
            "views":     int(vd.get("viewCount") or 0),
            "keywords":  vd.get("keywords", [])[:10],   # cap to save bandwidth
            "is_live":   vd.get("isLiveContent", False),
            "thumbnails":  sorted_thumbs,
            "thumbnail":   sorted_thumbs[0]["url"] if sorted_thumbs else "",
        }
        _cache_long.set(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Up-Next  (watch queue / radio for a song)
# ─────────────────────────────────────────────────────────────

@app.get("/upnext/{video_id}")
async def get_upnext(
    video_id: str,
    limit: int = Query(20, ge=5, le=50),
    force_refresh: bool = Query(False, description="Force a new queue even if one exists"),
):
    """
    Returns an Up-Next queue for the given video.

    Behaviour:
    • If a queue already exists for this video_id and it is < 2 hours old
      AND force_refresh is False → return the cached queue unchanged.
      This lets the frontend advance through the queue without triggering
      a new fetch for every track played from the queue.
    • Pass force_refresh=true (or the user manually picks a new song)
      to reset.
    """
    now = time.time()

    # Return existing queue if still valid
    existing = _upnext_store.get(video_id)
    if existing and not force_refresh:
        age = now - existing.get("created_at", 0)
        if age < _UPNEXT_TTL:
            return existing

    try:
        # get_watch_playlist returns {"tracks": [...], "lyrics": "...", ...}
        raw = await run(get_ytm().get_watch_playlist, video_id, None, limit)
        tracks_raw = raw.get("tracks") or []

        # Remove the first item if it's the song itself
        if tracks_raw and tracks_raw[0].get("videoId") == video_id:
            tracks_raw = tracks_raw[1:]

        tracks = [norm_track(t) for t in tracks_raw if t.get("videoId")]

        queue = {
            "origin_video_id": video_id,
            "tracks":          tracks,
            "count":           len(tracks),
            "created_at":      now,
        }
        _upnext_store[video_id] = queue

        # Prune store if it gets too large (keep last 100 queues)
        if len(_upnext_store) > 100:
            oldest_key = next(iter(_upnext_store))
            del _upnext_store[oldest_key]

        return queue
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.delete("/upnext/{video_id}", tags=["upnext"])
async def reset_upnext(video_id: str):
    """
    Explicitly clear the Up-Next queue for a video.
    Call this when the user manually selects a different song.
    """
    _upnext_store.pop(video_id, None)
    return {"cleared": video_id}


# ─────────────────────────────────────────────────────────────
#  Playlist
# ─────────────────────────────────────────────────────────────

@app.get("/playlist/{playlist_id}")
async def get_playlist(
    playlist_id: str,
    limit: int = 100,
    related: bool = False,
    suggestions_limit: int = 0,
):
    cache_key = f"playlist:{playlist_id}:{limit}"
    cached = _cache_medium.get(cache_key)
    if cached is not None:
        return cached

    try:
        data = await run(get_ytm().get_playlist, playlist_id, limit, related, suggestions_limit)
        # Fix track thumbnails
        for t in data.get("tracks") or []:
            if isinstance(t, dict):
                t["thumbnails"] = best_thumbnails_list(t.get("thumbnails") or [])
                t["thumbnail"]  = t["thumbnails"][0]["url"] if t["thumbnails"] else ""
        data["thumbnails"] = best_thumbnails_list(data.get("thumbnails") or [])
        data["thumbnail"]  = data["thumbnails"][0]["url"] if data["thumbnails"] else ""
        _cache_medium.set(cache_key, data)
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Watch playlist  (raw — used internally by /upnext)
# ─────────────────────────────────────────────────────────────

@app.get("/watch_playlist")
async def get_watch_playlist(
    video_id: Optional[str] = None,
    playlist_id: Optional[str] = None,
    limit: int = 25,
    params: Optional[str] = None,
):
    try:
        data = await run(get_ytm().get_watch_playlist, video_id, playlist_id, limit, params)
        for t in data.get("tracks") or []:
            if isinstance(t, dict):
                t["thumbnails"] = best_thumbnails_list(t.get("thumbnails") or [])
                t["thumbnail"]  = t["thumbnails"][0]["url"] if t["thumbnails"] else ""
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Home feed
# ─────────────────────────────────────────────────────────────

@app.get("/home")
async def get_home(limit: int = Query(6, ge=1, le=15)):
    """Home feed — returns shelves, each with a title and normalised tracks."""
    cache_key = f"home:{limit}"
    cached = _cache_medium.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = await run(get_ytm().get_home, limit)
        shelves = []
        for shelf in (raw or []):
            if not isinstance(shelf, dict):
                continue
            contents = []
            for item in (shelf.get("contents") or []):
                if not isinstance(item, dict):
                    continue
                if item.get("videoId"):
                    contents.append(norm_track(item))
                else:
                    # Could be a playlist/album card in the shelf
                    thumbs = best_thumbnails_list(item.get("thumbnails") or [])
                    item["thumbnails"] = thumbs
                    item["thumbnail"]  = thumbs[0]["url"] if thumbs else ""
                    contents.append(item)
            if contents:
                shelves.append({"title": shelf.get("title", "For You"), "contents": contents})

        _cache_medium.set(cache_key, shelves)
        return shelves
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Explore
# ─────────────────────────────────────────────────────────────

@app.get("/explore")
async def get_explore():
    cached = _cache_medium.get("explore")
    if cached is not None:
        return cached
    try:
        data = await run(get_ytm().get_explore)
        _cache_medium.set("explore", data)
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Charts  (fixed — extracts songs, videos, artists from raw dict)
# ─────────────────────────────────────────────────────────────

@app.get("/charts")
async def get_charts(country: str = "ZZ"):
    """
    Music charts, normalised.
    Returns: { songs, videos, artists, trending }  — each is a list of items.
    """
    cache_key = f"charts:{country}"
    cached = _cache_medium.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = await run(get_ytm().get_charts, country)

        def extract_section(section) -> list:
            if not section:
                return []
            if isinstance(section, list):
                items = section
            elif isinstance(section, dict):
                items = section.get("items") or section.get("results") or []
            else:
                return []
            out = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                thumbs = best_thumbnails_list(item.get("thumbnails") or [])
                item["thumbnails"] = thumbs
                item["thumbnail"]  = thumbs[0]["url"] if thumbs else ""
                out.append(item)
            return out

        result = {
            "country":  country,
            "songs":    [norm_track(t) for t in extract_section(raw.get("songs"))],
            "videos":   [norm_track(t) for t in extract_section(raw.get("videos"))
                         if isinstance(t, dict) and t.get("videoId")],
            "artists":  [norm_artist_result(a) for a in extract_section(raw.get("artists"))],
            "trending": [norm_track(t) for t in extract_section(raw.get("trending"))
                         if isinstance(t, dict) and t.get("videoId")],
        }

        _cache_medium.set(cache_key, result)
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Mood categories  (fixed)
# ─────────────────────────────────────────────────────────────

@app.get("/mood_categories")
async def get_mood_categories():
    """
    Returns mood/genre categories as a flat list:
    [ { title, params, thumbnails, thumbnail }, ... ]
    """
    cached = _cache_medium.get("mood_categories")
    if cached is not None:
        return cached

    try:
        raw = await run(get_ytm().get_mood_categories)
        """
        ytmusicapi returns a dict of sections like:
        { "Moods & moments": [ {title, params, thumbnails}, ... ],
          "Genres":          [ ... ] }
        We flatten everything into one list.
        """
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

        _cache_medium.set("mood_categories", categories)
        return categories
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Mood playlists  (fixed)
# ─────────────────────────────────────────────────────────────

@app.get("/mood_playlists/{params}")
async def get_mood_playlists(params: str):
    """
    Playlists for a given mood params string.
    Returns a flat list of playlist cards with thumbnails.
    """
    cache_key = f"mood_playlists:{params}"
    cached = _cache_medium.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = await run(get_ytm().get_mood_playlists, params)
        """
        ytmusicapi returns:
        [ { title: "section title", contents: [ {title, browseId, thumbnails}, ... ] }, ... ]
        We flatten all contents across sections.
        """
        playlists: list[dict] = []
        if isinstance(raw, list):
            for section in raw:
                if not isinstance(section, dict):
                    continue
                contents = section.get("contents") or section.get("playlists") or []
                if isinstance(contents, list):
                    for item in contents:
                        if not isinstance(item, dict):
                            continue
                        thumbs = best_thumbnails_list(item.get("thumbnails") or [])
                        playlists.append({
                            "browseId":  item.get("playlistId") or item.get("browseId", ""),
                            "title":     item.get("title", ""),
                            "subtitle":  item.get("subtitle", ""),
                            "thumbnails": thumbs,
                            "thumbnail":  thumbs[0]["url"] if thumbs else "",
                        })
                elif not contents:
                    # section itself is the item
                    thumbs = best_thumbnails_list(section.get("thumbnails") or [])
                    playlists.append({
                        "browseId":  section.get("playlistId") or section.get("browseId", ""),
                        "title":     section.get("title", ""),
                        "subtitle":  section.get("subtitle", ""),
                        "thumbnails": thumbs,
                        "thumbnail":  thumbs[0]["url"] if thumbs else "",
                    })

        _cache_medium.set(cache_key, playlists)
        return playlists
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Lyrics
# ─────────────────────────────────────────────────────────────

@app.get("/lyrics/{browse_id}")
async def get_lyrics(browse_id: str):
    cache_key = f"lyrics:{browse_id}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached
    try:
        data = await run(get_ytm().get_lyrics, browse_id)
        _cache_long.set(cache_key, data)
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  User endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/user/{channel_id}")
async def get_user(channel_id: str):
    cache_key = f"user:{channel_id}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached
    try:
        data = await run(get_ytm().get_user, channel_id)
        _cache_long.set(cache_key, data)
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/user_playlists/{channel_id}")
async def get_user_playlists(channel_id: str, params: Optional[str] = None):
    try:
        data = await run(get_ytm().get_user_playlists, channel_id, params)
        return data
    except Exception as e:
        raise HTTPException(500, detail=str(e))
