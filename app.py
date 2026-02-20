from fastapi import FastAPI, HTTPException, Query
from ytmusicapi import YTMusic
from typing import Optional, List

app = FastAPI(
    title="YouTube Music API Wrapper",
    description="A FastAPI wrapper for unauthenticated ytmusicapi features.",
    version="1.0.0"
)

ytmusic = YTMusic()  # Initialize without authentication

# Error handler for robustness
def handle_ytmusic_error(e: Exception):
    raise HTTPException(status_code=500, detail=f"YouTube Music API error: {str(e)}")

@app.get("/search")
async def search(
    query: str,
    filter: Optional[str] = Query(None, description="Filter: songs, videos, albums, artists, playlists, podcasts"),
    scope: Optional[str] = Query(None, description="Scope: library, uploads (though unauthenticated limits apply)"),
    limit: int = Query(20, ge=1),
    ignore_spelling: bool = False
):
    """
    Perform a search with metadata (title, artist, thumbnails, etc.).
    Use 'filter' for specific types like podcasts, videos, albums, artists.
    """
    try:
        return ytmusic.search(query, filter, scope, limit, ignore_spelling)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/search_suggestions")
async def search_suggestions(
    query: str,
    detailed_runs: bool = False
):
    """Get search suggestions."""
    try:
        return ytmusic.get_search_suggestions(query, detailed_runs)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/artist/{artist_id}")
async def get_artist(artist_id: str, filter: Optional[str] = None, limit: int = 20, offset: Optional[int] = None):
    """Get artist details with metadata (thumbnails, name, etc.)."""
    try:
        return ytmusic.get_artist(artist_id, filter, limit, offset)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/album/{album_id}")
async def get_album(album_id: str):
    """Get album details with metadata (thumbnails, tracks, etc.)."""
    try:
        return ytmusic.get_album(album_id)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/song/{video_id}")
async def get_song(video_id: str):
    """Get song details with metadata (thumbnails, title, etc.)."""
    try:
        return ytmusic.get_song(video_id)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/playlist/{playlist_id}")
async def get_playlist(playlist_id: str, limit: int = 100, related: bool = False, suggestions_limit: int = 0):
    """Get public playlist details with metadata."""
    try:
        return ytmusic.get_playlist(playlist_id, limit, related, suggestions_limit)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/watch_playlist")
async def get_watch_playlist(
    video_id: Optional[str] = None,
    playlist_id: Optional[str] = None,
    limit: int = 25,
    params: Optional[str] = None
):
    """Get watch playlist with metadata."""
    try:
        return ytmusic.get_watch_playlist(video_id, playlist_id, limit, params)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/lyrics/{browse_id}")
async def get_lyrics(browse_id: str):
    """Get lyrics for a song."""
    try:
        return ytmusic.get_lyrics(browse_id)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/user/{channel_id}")
async def get_user(channel_id: str):
    """Get public user profile."""
    try:
        return ytmusic.get_user(channel_id)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/user_playlists/{channel_id}")
async def get_user_playlists(channel_id: str, params: Optional[str] = None):
    """Get public user playlists."""
    try:
        return ytmusic.get_user_playlists(channel_id, params)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/home")
async def get_home(limit: int = 5):
    """Get home feed."""
    try:
        return ytmusic.get_home(limit)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/explore")
async def get_explore():
    """Get explore page content."""
    try:
        return ytmusic.get_explore()
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/charts")
async def get_charts(country: str = "ZZ"):
    """Get music charts for a country."""
    try:
        return ytmusic.get_charts(country)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/mood_categories")
async def get_mood_categories():
    """Get mood categories."""
    try:
        return ytmusic.get_mood_categories()
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/mood_playlists/{params}")
async def get_mood_playlists(params: str):
    """Get mood playlists by params."""
    try:
        return ytmusic.get_mood_playlists(params)
    except Exception as e:
        handle_ytmusic_error(e)

@app.get("/stream/{video_id}")
async def get_stream(video_id: str):
    """
    Instead of streaming, return video ID with metadata and thumbnails.
    Metadata includes title, artist, duration, views, etc., from song details.
    """
    try:
        song_data = ytmusic.get_song(video_id)
        if 'videoDetails' not in song_data:
            raise ValueError("No video details found.")
        
        video_details = song_data['videoDetails']
        metadata = {
            "title": video_details.get('title'),
            "artist": video_details.get('author'),
            "duration_seconds": int(video_details.get('lengthSeconds', 0)),
            "views": int(video_details.get('viewCount', 0)),
            "keywords": video_details.get('keywords', []),
            # Add more as needed from videoDetails
        }
        thumbnails = song_data.get('thumbnail', {}).get('thumbnails', [])
        
        return {
            "videoId": video_id,
            "metadata": metadata,
            "thumbnails": thumbnails
        }
    except Exception as e:
        handle_ytmusic_error(e)
