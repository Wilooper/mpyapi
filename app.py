"""
YouTube Music API Wrapper — v4  (Render 512 MB free-tier optimised)
====================================================================
Changes over v2/v3:
  • /search          — proper offset-based pagination { results, count, total, hasMore }
  • /podcast/{id}    — NEW: full podcast page with normalised episodes list
  • /artist/{id}/songs — NEW: all songs by artist (follows ytmusicapi params token)
  • /now_playing/{id}  — NEW: stream metadata + related in one call
  • /related_songs/{id}— NEW: related tracks via watch-playlist
  • /lyrics_by_video/{id} — NEW: get lyrics without separate browseId lookup
  • /trending         — NEW: trending songs from charts (country-aware)
  • /top_playlists    — NEW: featured/top playlists
  • /playlist/{id}    — handles VL-prefix IDs correctly, returns normalised tracks
  • Podcast search normalisation — extracts podcastId / browseId / channelId
  • Artist page       — includes songs.params token so frontend can paginate
  • Stream endpoint   — adds url + audio_url alias so frontend download works
  • Memory budget     — cache sizes trimmed for 512 MB; thread pool stays at 4
  • All errors return JSON 500 with detail (never crash the server)
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
    title="YouTube Music API — v4",
    description="FastAPI wrapper for unauthenticated ytmusicapi. Render 512 MB optimised.",
    version="4.0.0",
)

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "DELETE"],
    allow_headers=["*"],
)

# Thread pool — keep small on 512 MB RAM (ytmusicapi is I/O-bound, not CPU)
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
    """Run a blocking ytmusicapi call in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


# ─────────────────────────────────────────────────────────────
#  TTL Cache  (simple LRU with expiry — no external deps)
# ─────────────────────────────────────────────────────────────

class TTLCache:
    """Thread-safe in-memory LRU cache with per-entry TTL."""

    def __init__(self, maxsize: int = 128, ttl: int = 300):
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


# Trimmed sizes for 512 MB RAM
_cache_short  = TTLCache(maxsize=64,  ttl=120)   # 2 min  — search, suggestions
_cache_medium = TTLCache(maxsize=96,  ttl=600)   # 10 min — home, charts, playlists
_cache_long   = TTLCache(maxsize=192, ttl=3600)  # 1 hr   — artist, album, song, podcast

# ─────────────────────────────────────────────────────────────
#  Up-Next queue store  (in-memory, no Redis needed)
# ─────────────────────────────────────────────────────────────

_upnext_store: Dict[str, Dict] = {}
_UPNEXT_TTL = 7200  # 2 hours


# ─────────────────────────────────────────────────────────────
#  Thumbnail helpers
# ─────────────────────────────────────────────────────────────

def best_thumbnail(thumbnails: list | None) -> str:
    """Return URL of the highest-resolution thumbnail."""
    if not thumbnails:
        return ""
    try:
        s = sorted(thumbnails, key=lambda t: (t.get("width", 0) or 0, t.get("height", 0) or 0), reverse=True)
        return s[0].get("url", "") if s else ""
    except Exception:
        return thumbnails[-1].get("url", "") if thumbnails else ""


def best_thumbnails_list(thumbnails: list | None) -> list:
    """Return thumbnails sorted best-first, URL-deduplicated."""
    if not thumbnails:
        return []
    try:
        seen, out = set(), []
        for t in sorted(thumbnails, key=lambda t: (t.get("width", 0) or 0), reverse=True):
            url = t.get("url", "")
            if url and url not in seen:
                seen.add(url)
                out.append(t)
        return out
    except Exception:
        return thumbnails


# ─────────────────────────────────────────────────────────────
#  Normalisation helpers
# ─────────────────────────────────────────────────────────────

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
                out.append({"name": a.get("name", ""), "id": a.get("id", "") or a.get("browseId", "")})
        return out
    return []


def norm_track(t: dict) -> dict:
    """Normalise a track/song dict to a consistent shape the frontend expects."""
    raw_thumbs = t.get("thumbnails") or t.get("thumbnail") or []
    if isinstance(raw_thumbs, str):
        raw_thumbs = [{"url": raw_thumbs, "width": 0, "height": 0}]
    thumbs = best_thumbnails_list(raw_thumbs)
    album = t.get("album")
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
    thumbs = best_thumbnails_list(p.get("thumbnails") or [])
    # ytmusicapi uses browseId OR podcastId; handle all variants
    browse_id = (
        p.get("browseId") or
        p.get("podcastId") or
        p.get("channelId") or
        ""
    )
    author = p.get("author", "") or ", ".join(
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
        pass


# ─────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "4.0.0"}


# ─────────────────────────────────────────────────────────────
#  Search  (v4 — paginated with offset)
# ─────────────────────────────────────────────────────────────

@app.get("/search")
async def search(
    query: str,
    filter: Optional[str] = Query(None, description="songs|videos|albums|artists|playlists|podcasts"),
    scope: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    ignore_spelling: bool = False,
):
    """
    Search YouTube Music.
    Returns: { results, count, total, hasMore, offset, limit }
    All items are normalised (thumbnail string at top level).
    """
    # ytmusicapi doesn't support offset natively — we fetch limit+offset and slice
    # Cap total fetch at 50 to stay within RAM budget
    fetch_limit = min(offset + limit, 50)
    cache_key = f"search:{query}:{filter}:{fetch_limit}"
    cached = _cache_short.get(cache_key)

    if cached is None:
        try:
            raw = await run(get_ytm().search, query, filter, scope, fetch_limit, ignore_spelling)
            cached = norm_search_results(raw or [], filter)
            _cache_short.set(cache_key, cached)
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    page = cached[offset:offset + limit]
    return {
        "results": page,
        "count":   len(page),
        "total":   len(cached),
        "hasMore": len(cached) > offset + limit,
        "offset":  offset,
        "limit":   limit,
    }


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
async def get_artist(artist_id: str):
    """
    Full artist page.
    Thumbnails sorted best-first.
    songs.params token included so caller can fetch more songs.
    """
    cache_key = f"artist:{artist_id}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached

    try:
        data = await run(get_ytm().get_artist, artist_id)
        if not data:
            raise HTTPException(404, "Artist not found")

        # Top-level thumbnails
        data["thumbnails"] = best_thumbnails_list(data.get("thumbnails") or [])
        data["thumbnail"]  = data["thumbnails"][0]["url"] if data["thumbnails"] else ""

        # Nested sections
        for section in ("songs", "albums", "singles", "videos", "related"):
            section_data = data.get(section, {})
            if not isinstance(section_data, dict):
                continue
            results = section_data.get("results") or section_data.get("items") or []
            normalised = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                item["thumbnails"] = best_thumbnails_list(item.get("thumbnails") or [])
                item["thumbnail"]  = item["thumbnails"][0]["url"] if item["thumbnails"] else ""
                normalised.append(item)
            section_data["results"] = normalised
            # Preserve the params token — frontend uses it to load ALL songs
            # data["songs"]["params"] is the key used by get_artist_albums / browse_artist
            data[section] = section_data

        _cache_long.set(cache_key, data)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Artist — ALL songs  (NEW)
# ─────────────────────────────────────────────────────────────

@app.get("/artist/{artist_id}/songs")
async def get_artist_songs(
    artist_id: str,
    limit: int = Query(100, ge=1, le=200),
):
    """
    All songs by an artist.
    Returns: { artistId, name, songs: [norm_track], total }

    Strategy:
      1. Fetch artist page to get the songs.params token.
      2. Use get_artist_albums (browse) to follow the "all songs" link.
      3. Falls back to a songs-filter search by artist name if step 2 fails.
    """
    cache_key = f"artist_songs:{artist_id}:{limit}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached

    try:
        ytm = get_ytm()
        artist_data = await run(ytm.get_artist, artist_id)
        if not artist_data:
            raise HTTPException(404, "Artist not found")

        artist_name = artist_data.get("name", "")
        songs_section = artist_data.get("songs", {})
        params_token = None
        if isinstance(songs_section, dict):
            params_token = songs_section.get("params") or songs_section.get("browseId")

        songs_raw: list = []

        # Step 1: try get_artist_albums (follow "all songs" browse token)
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

        # Step 2: use watch-playlist / radio if browse token gave nothing
        if not songs_raw and isinstance(songs_section, dict):
            initial = songs_section.get("results") or songs_section.get("items") or []
            songs_raw = list(initial)

        # Step 3: search fallback
        if not songs_raw and artist_name:
            try:
                search_raw = await run(ytm.search, artist_name, "songs", None, min(limit, 50), False)
                if isinstance(search_raw, list):
                    songs_raw = search_raw
            except Exception:
                pass

        tracks = [norm_track(t) for t in songs_raw if isinstance(t, dict) and t.get("videoId")]
        # Deduplicate by videoId
        seen, deduped = set(), []
        for t in tracks:
            vid = t.get("videoId", "")
            if vid and vid not in seen:
                seen.add(vid)
                deduped.append(t)

        result = {
            "artistId": artist_id,
            "name":     artist_name,
            "songs":    deduped[:limit],
            "total":    len(deduped),
        }
        _cache_long.set(cache_key, result)
        return result
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
        data["tracks"] = [norm_track(t) for t in (data.get("tracks") or [])]
        _cache_long.set(cache_key, data)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Song metadata
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
#  Stream metadata  (used by frontend for YouTube IFrame + download)
# ─────────────────────────────────────────────────────────────

@app.get("/stream/{video_id}")
async def get_stream(video_id: str):
    """
    Returns metadata + highest-quality thumbnails for a video.
    Includes url / audio_url aliases so the frontend download works.
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
        thumb_url = sorted_thumbs[0]["url"] if sorted_thumbs else ""

        # Canonical YouTube watch URL — frontend embeds via IFrame player
        watch_url = f"https://www.youtube.com/watch?v={video_id}"

        result = {
            "video_id":        video_id,
            "videoId":         video_id,
            # url / audio_url / stream_url — all same value; frontend tries all three
            "url":             watch_url,
            "audio_url":       watch_url,
            "stream_url":      watch_url,
            "title":           vd.get("title", ""),
            "artist":          vd.get("author", ""),
            "channel_id":      vd.get("channelId", ""),
            "duration_seconds": int(vd.get("lengthSeconds") or 0),
            "views":           int(vd.get("viewCount") or 0),
            "keywords":        vd.get("keywords", [])[:10],
            "is_live":         vd.get("isLiveContent", False),
            "thumbnails":      sorted_thumbs,
            "thumbnail":       thumb_url,
        }
        _cache_long.set(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Now Playing  (NEW — stream metadata + related in one call)
# ─────────────────────────────────────────────────────────────

@app.get("/now_playing/{video_id}")
async def now_playing(
    video_id: str,
    related_limit: int = Query(10, ge=1, le=30),
):
    """
    One-shot endpoint used by the frontend player.
    Returns: { videoId, stream: {…}, related: [norm_track] }
    Avoids two round-trips from the client.
    """
    cache_key = f"now_playing:{video_id}:{related_limit}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached

    ytm = get_ytm()

    # Fetch in parallel using asyncio.gather
    stream_task  = run(ytm.get_song, video_id)
    related_task = run(ytm.get_watch_playlist, video_id, None, related_limit + 1)

    try:
        song_data, watch_data = await asyncio.gather(stream_task, related_task, return_exceptions=True)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    # --- Stream data ---
    stream: dict = {}
    if isinstance(song_data, dict):
        vd = song_data.get("videoDetails", {})
        raw_thumbs = (
            vd.get("thumbnail", {}).get("thumbnails") or
            song_data.get("thumbnail", {}).get("thumbnails") or []
        )
        sorted_thumbs = best_thumbnails_list(raw_thumbs)
        thumb_url = sorted_thumbs[0]["url"] if sorted_thumbs else ""
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        stream = {
            "videoId":         video_id,
            "url":             watch_url,
            "audio_url":       watch_url,
            "title":           vd.get("title", ""),
            "artist":          vd.get("author", ""),
            "duration_seconds": int(vd.get("lengthSeconds") or 0),
            "thumbnails":      sorted_thumbs,
            "thumbnail":       thumb_url,
        }

    # --- Related tracks ---
    related: list = []
    if isinstance(watch_data, dict):
        tracks_raw = watch_data.get("tracks") or []
        # Skip the first item if it is the current song itself
        if tracks_raw and tracks_raw[0].get("videoId") == video_id:
            tracks_raw = tracks_raw[1:]
        related = [norm_track(t) for t in tracks_raw[:related_limit] if t.get("videoId")]

    result = {
        "videoId": video_id,
        "stream":  stream,
        "related": related,
    }
    _cache_long.set(cache_key, result, ttl=1800)  # 30 min
    return result


# ─────────────────────────────────────────────────────────────
#  Related songs  (NEW)
# ─────────────────────────────────────────────────────────────

@app.get("/related_songs/{video_id}")
async def get_related_songs(
    video_id: str,
    limit: int = Query(15, ge=1, le=50),
):
    """
    Related tracks via watch-playlist radio.
    Returns: { videoId, tracks: [norm_track], count }
    """
    cache_key = f"related:{video_id}:{limit}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached
    try:
        raw = await run(get_ytm().get_watch_playlist, video_id, None, limit + 1)
        tracks_raw = raw.get("tracks") or []
        if tracks_raw and tracks_raw[0].get("videoId") == video_id:
            tracks_raw = tracks_raw[1:]
        tracks = [norm_track(t) for t in tracks_raw[:limit] if t.get("videoId")]
        result = {"videoId": video_id, "tracks": tracks, "count": len(tracks)}
        _cache_long.set(cache_key, result, ttl=1800)
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Up-Next  (watch queue / radio for a song)
# ─────────────────────────────────────────────────────────────

@app.get("/upnext/{video_id}")
async def get_upnext(
    video_id: str,
    limit: int = Query(20, ge=5, le=50),
    force_refresh: bool = Query(False),
):
    """
    Returns an Up-Next queue for the given video.
    Cached in-memory for 2 hours so advancing the queue doesn't re-fetch.
    Pass force_refresh=true when user manually picks a new song.
    """
    now = time.time()
    existing = _upnext_store.get(video_id)
    if existing and not force_refresh:
        if now - existing.get("created_at", 0) < _UPNEXT_TTL:
            return existing

    try:
        raw = await run(get_ytm().get_watch_playlist, video_id, None, limit)
        tracks_raw = raw.get("tracks") or []
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
        # Keep store bounded
        if len(_upnext_store) > 100:
            oldest = next(iter(_upnext_store))
            del _upnext_store[oldest]
        return queue
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.delete("/upnext/{video_id}")
async def reset_upnext(video_id: str):
    """Explicitly clear the Up-Next queue for a video."""
    _upnext_store.pop(video_id, None)
    return {"cleared": video_id}


# ─────────────────────────────────────────────────────────────
#  Playlist  (handles VL-prefix IDs)
# ─────────────────────────────────────────────────────────────

@app.get("/playlist/{playlist_id}")
async def get_playlist(
    playlist_id: str,
    limit: int = Query(100, ge=1, le=500),
    related: bool = False,
    suggestions_limit: int = 0,
):
    """
    Full playlist with normalised tracks.
    Handles YouTube playlist IDs that start with VL (strips the prefix).
    """
    # ytmusicapi expects IDs WITHOUT the VL prefix
    clean_id = playlist_id.lstrip("VL") if playlist_id.startswith("VL") else playlist_id

    cache_key = f"playlist:{clean_id}:{limit}"
    cached = _cache_medium.get(cache_key)
    if cached is not None:
        return cached

    try:
        data = await run(get_ytm().get_playlist, clean_id, limit, related, suggestions_limit)
        if not data:
            raise HTTPException(404, "Playlist not found")

        # Normalise tracks
        tracks = data.get("tracks") or []
        data["tracks"] = [norm_track(t) for t in tracks if isinstance(t, dict)]

        # Normalise playlist-level thumbnails
        data["thumbnails"] = best_thumbnails_list(data.get("thumbnails") or [])
        data["thumbnail"]  = data["thumbnails"][0]["url"] if data["thumbnails"] else ""

        _cache_medium.set(cache_key, data)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Podcast  (NEW — full podcast page with episodes)
# ─────────────────────────────────────────────────────────────

@app.get("/podcast/{podcast_id}")
async def get_podcast(
    podcast_id: str,
    limit: int = Query(50, ge=1, le=200),
):
    """
    Full podcast page.
    Returns: { podcastId, title, author, description, thumbnail, thumbnails, episodes, total }
    Each episode: { videoId, title, artist, thumbnail, duration, date, description }

    ytmusicapi `get_podcast` is available in ≥ 1.7.0.  Falls back to
    `get_playlist` (podcasts are playlists internally) if it fails.
    """
    # Strip common prefix variants
    clean_id = podcast_id
    for prefix in ("VLMPSPPLxx", "MPSPPL", "VL"):
        if clean_id.startswith(prefix) and not clean_id.startswith("VLMPSPPLxx"):
            pass  # only strip VL if it's truly a VL prefix
    if clean_id.startswith("VL") and not clean_id.startswith("VLM"):
        clean_id = clean_id[2:]

    cache_key = f"podcast:{clean_id}:{limit}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached

    ytm = get_ytm()
    episodes_raw: list = []
    meta: dict = {}

    # Attempt 1: native get_podcast (ytmusicapi ≥ 1.7)
    try:
        data = await run(ytm.get_podcast, clean_id)
        if isinstance(data, dict) and data:
            meta = data
            # Episodes can be under "episodes" or "tracks"
            episodes_raw = data.get("episodes") or data.get("tracks") or []
    except Exception:
        pass

    # Attempt 2: treat as playlist (podcasts are playlists in YT Music)
    if not episodes_raw:
        try:
            data = await run(ytm.get_playlist, clean_id, limit)
            if isinstance(data, dict) and data:
                meta = meta or data
                episodes_raw = data.get("tracks") or []
        except Exception:
            pass

    # Attempt 3: also try with original ID if clean_id differs
    if not episodes_raw and clean_id != podcast_id:
        try:
            data = await run(ytm.get_podcast, podcast_id)
            if isinstance(data, dict):
                meta = meta or data
                episodes_raw = data.get("episodes") or data.get("tracks") or []
        except Exception:
            pass
        if not episodes_raw:
            try:
                data = await run(ytm.get_playlist, podcast_id, limit)
                if isinstance(data, dict):
                    meta = meta or data
                    episodes_raw = data.get("tracks") or []
            except Exception:
                pass

    if not meta and not episodes_raw:
        raise HTTPException(404, "Podcast not found")

    # Normalise episodes
    def norm_episode(ep: dict) -> dict:
        raw_t = ep.get("thumbnails") or ep.get("thumbnail") or []
        if isinstance(raw_t, str):
            raw_t = [{"url": raw_t, "width": 0}]
        thumbs = best_thumbnails_list(raw_t)
        thumb  = thumbs[0]["url"] if thumbs else ""

        # duration can be seconds (int) or "M:SS" string
        dur = ep.get("duration") or ep.get("durationSeconds") or ""
        if isinstance(dur, int) and dur > 0:
            m, s = divmod(dur, 60)
            h, m = divmod(m, 60)
            dur = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

        artists = ep.get("artists") or ep.get("author") or ""
        if isinstance(artists, list):
            artists = ", ".join(a.get("name", "") if isinstance(a, dict) else str(a) for a in artists)

        return {
            "videoId":     ep.get("videoId", ""),
            "title":       ep.get("title", ""),
            "artist":      artists or "",
            "thumbnail":   thumb,
            "thumbnails":  thumbs,
            "duration":    str(dur),
            "date":        ep.get("date") or ep.get("publishedTime") or "",
            "description": ep.get("description") or ep.get("shortDescription") or "",
        }

    episodes = [
        norm_episode(ep)
        for ep in episodes_raw
        if isinstance(ep, dict) and ep.get("videoId")
    ][:limit]

    # Podcast-level thumbnail
    meta_thumbs = best_thumbnails_list(meta.get("thumbnails") or [])
    meta_thumb  = meta_thumbs[0]["url"] if meta_thumbs else ""

    # Author / channel
    author = (
        meta.get("author") or
        meta.get("channel", {}).get("name", "") if isinstance(meta.get("channel"), dict) else "" or
        meta.get("artist") or
        ""
    )
    if isinstance(author, dict):
        author = author.get("name", "")

    result = {
        "podcastId":   clean_id,
        "title":       meta.get("title", ""),
        "author":      author,
        "description": meta.get("description") or meta.get("shortDescription") or "",
        "thumbnail":   meta_thumb,
        "thumbnails":  meta_thumbs,
        "episodes":    episodes,
        "total":       len(episodes),
    }

    _cache_long.set(cache_key, result)
    return result


# ─────────────────────────────────────────────────────────────
#  Watch playlist  (raw — used internally)
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
#  Lyrics  (by browseId — for when you already have it)
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
#  Lyrics by video ID  (NEW — no separate browseId needed)
# ─────────────────────────────────────────────────────────────

@app.get("/lyrics_by_video/{video_id}")
async def get_lyrics_by_video(video_id: str):
    """
    Get lyrics for a video without a separate browseId lookup.
    Returns: { lyricsId, lyrics } or { lyricsId: null, error: "..." }
    """
    cache_key = f"lyrics_vid:{video_id}"
    cached = _cache_long.get(cache_key)
    if cached is not None:
        return cached

    ytm = get_ytm()

    # Step 1: get watch playlist to extract lyricsId
    try:
        watch = await run(ytm.get_watch_playlist, video_id, None, 1)
        lyrics_id = watch.get("lyrics")
    except Exception:
        lyrics_id = None

    if not lyrics_id:
        result = {"lyricsId": None, "lyrics": None, "error": "Lyrics not available"}
        _cache_long.set(cache_key, result, ttl=600)
        return result

    # Step 2: fetch actual lyrics
    try:
        data = await run(ytm.get_lyrics, lyrics_id)
        result = {
            "lyricsId": lyrics_id,
            "lyrics":   data.get("lyrics") if isinstance(data, dict) else data,
            "source":   data.get("source", "") if isinstance(data, dict) else "",
        }
        _cache_long.set(cache_key, result)
        return result
    except Exception as e:
        result = {"lyricsId": lyrics_id, "lyrics": None, "error": str(e)}
        _cache_long.set(cache_key, result, ttl=300)
        return result


# ─────────────────────────────────────────────────────────────
#  Home feed
# ─────────────────────────────────────────────────────────────

@app.get("/home")
async def get_home(limit: int = Query(6, ge=1, le=15)):
    """Home feed — shelves with normalised tracks."""
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
#  Charts  (songs, videos, artists, trending sections)
# ─────────────────────────────────────────────────────────────

@app.get("/charts")
async def get_charts(country: str = "ZZ"):
    """Music charts. Returns: { country, songs, videos, artists, trending }"""
    cache_key = f"charts:{country}"
    cached = _cache_medium.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = await run(get_ytm().get_charts, country)

        def extract_section(section) -> list:
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

        result = {
            "country":  country,
            "songs":    [norm_track(t) for t in extract_section(raw.get("songs"))],
            "videos":   [norm_track(t) for t in extract_section(raw.get("videos")) if t.get("videoId")],
            "artists":  [norm_artist_result(a) for a in extract_section(raw.get("artists"))],
            "trending": [norm_track(t) for t in extract_section(raw.get("trending")) if t.get("videoId")],
        }
        _cache_medium.set(cache_key, result)
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  Trending  (NEW — convenience wrapper over /charts)
# ─────────────────────────────────────────────────────────────

@app.get("/trending")
async def get_trending(
    country: str = Query("ZZ"),
    limit:   int = Query(20, ge=1, le=50),
):
    """
    Trending songs for a country.
    Returns: { country, trending: [norm_track], count }
    Pulls from /charts and returns the trending section.
    """
    # Reuse charts cache
    cache_key_charts = f"charts:{country}"
    charts_cached = _cache_medium.get(cache_key_charts)

    if charts_cached:
        trending = charts_cached.get("trending") or charts_cached.get("songs") or []
    else:
        try:
            raw = await run(get_ytm().get_charts, country)
            trending_raw = (
                raw.get("trending") or raw.get("songs") or []
            )
            if isinstance(trending_raw, dict):
                trending_raw = trending_raw.get("items") or trending_raw.get("results") or []
            trending = [norm_track(t) for t in trending_raw if isinstance(t, dict) and t.get("videoId")]
            # Also cache the full charts result
            charts_result = {
                "country":  country,
                "songs":    [norm_track(t) for t in (raw.get("songs") or {}).get("items", []) if isinstance(t, dict)],
                "videos":   [],
                "artists":  [],
                "trending": trending,
            }
            _cache_medium.set(cache_key_charts, charts_result)
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    result = {
        "country":  country,
        "trending": trending[:limit],
        "count":    len(trending[:limit]),
    }
    return result


# ─────────────────────────────────────────────────────────────
#  Top Playlists  (NEW)
# ─────────────────────────────────────────────────────────────

@app.get("/top_playlists")
async def get_top_playlists(
    country: str = Query("ZZ"),
    limit:   int = Query(16, ge=1, le=50),
):
    """
    Featured / top playlists. Pulls from mood_categories + charts.
    Returns a flat list of playlist cards: [ { browseId, title, thumbnail, … } ]
    """
    cache_key = f"top_playlists:{country}:{limit}"
    cached = _cache_medium.get(cache_key)
    if cached is not None:
        return cached

    ytm = get_ytm()
    playlists: list[dict] = []

    # Source 1: Charts — trending playlists from country
    try:
        charts_raw = await run(ytm.get_charts, country)
        for section_key in ("trending", "songs"):
            section = charts_raw.get(section_key, {})
            items = section if isinstance(section, list) else (
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

    # Source 2: Mood categories → first mood playlists batch
    if len(playlists) < limit:
        try:
            cats = await run(ytm.get_mood_categories)
            first_params = None
            if isinstance(cats, dict):
                for section_items in cats.values():
                    if isinstance(section_items, list) and section_items:
                        p = section_items[0].get("params")
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
                    contents = section.get("contents") or section.get("playlists") or []
                    for item in contents:
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
            seen.add(bid)
            deduped.append(p)

    result = deduped[:limit]
    _cache_medium.set(cache_key, result)
    return result


# ─────────────────────────────────────────────────────────────
#  Mood categories
# ─────────────────────────────────────────────────────────────

@app.get("/mood_categories")
async def get_mood_categories():
    """
    Mood/genre categories as a flat list.
    Returns: [ { title, params, section, thumbnail, thumbnails } ]
    """
    cached = _cache_medium.get("mood_categories")
    if cached is not None:
        return cached
    try:
        raw = await run(get_ytm().get_mood_categories)
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
#  Mood playlists
# ─────────────────────────────────────────────────────────────

@app.get("/mood_playlists/{params}")
async def get_mood_playlists(params: str):
    """
    Playlists for a given mood params string.
    Returns: [ { browseId, title, subtitle, thumbnail, thumbnails } ]
    """
    cache_key = f"mood_playlists:{params}"
    cached = _cache_medium.get(cache_key)
    if cached is not None:
        return cached
    try:
        raw = await run(get_ytm().get_mood_playlists, params)
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
        return playlists
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
