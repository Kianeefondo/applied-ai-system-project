import re
from typing import Dict, List, Optional, Tuple

Song = Dict[str, object]
PlaylistMap = Dict[str, List[Song]]

DEFAULT_PROFILE = {
    "name": "Default",
    "hype_min_energy": 7,
    "chill_max_energy": 3,
    "favorite_genre": "rock",
    "include_mixed": True,
}


def normalize_title(title: str) -> str:
    """Normalize a song title for comparisons."""
    if not isinstance(title, str):
        return ""
    return title.strip()


def normalize_artist(artist: str) -> str:
    """Normalize an artist name for comparisons."""
    if not artist:
        return ""
    return artist.strip().lower()


def normalize_genre(genre: str) -> str:
    """Normalize a genre name for comparisons."""
    return genre.lower().strip()


def normalize_song(raw: Song) -> Song:
    """Return a normalized song dict with expected keys."""
    title = normalize_title(str(raw.get("title", "")))
    artist = normalize_artist(str(raw.get("artist", "")))
    genre = normalize_genre(str(raw.get("genre", "")))
    energy = raw.get("energy", 0)

    if isinstance(energy, str):
        try:
            energy = int(energy)
        except ValueError:
            energy = 0

    tags = raw.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    tags = [str(tag).strip().lower() for tag in tags if tag]

    return {
        "title": title,
        "artist": artist,
        "genre": genre,
        "energy": energy,
        "tags": tags,
    }


def classify_song(song: Song, profile: Dict[str, object]) -> str:
    """Return a mood label given a song and user profile."""
    energy = song.get("energy", 0)
    genre = song.get("genre", "")
    title = song.get("title", "")

    hype_min_energy = profile.get("hype_min_energy", 7)
    chill_max_energy = profile.get("chill_max_energy", 3)
    favorite_genre = profile.get("favorite_genre", "")

    hype_keywords = ["rock", "punk", "party"]
    chill_keywords = ["lofi", "ambient", "sleep"]
    tags = song.get("tags", [])

    is_hype_keyword = any(k in genre for k in hype_keywords) or any(k in tag for tag in tags for k in hype_keywords)
    is_chill_keyword = any(k in genre for k in chill_keywords) or any(k in tag for tag in tags for k in chill_keywords)

    if genre == favorite_genre or energy >= hype_min_energy or is_hype_keyword:
        return "Hype"
    if energy <= chill_max_energy or is_chill_keyword:
        return "Chill"
    return "Mixed"


def build_playlists(songs: List[Song], profile: Dict[str, object]) -> PlaylistMap:
    """Group songs into playlists based on mood and profile."""
    playlists: PlaylistMap = {
        "Hype": [],
        "Chill": [],
        "Mixed": [],
    }

    for song in songs:
        normalized = normalize_song(song)
        mood = classify_song(normalized, profile)
        normalized["mood"] = mood
        playlists[mood].append(normalized)

    return playlists


def merge_playlists(a: PlaylistMap, b: PlaylistMap) -> PlaylistMap:
    """Merge two playlist maps into a new map."""
    merged: PlaylistMap = {}
    for key in set(list(a.keys()) + list(b.keys())):
        merged[key] = a.get(key, [])
        merged[key].extend(b.get(key, []))
    return merged


def compute_playlist_stats(playlists: PlaylistMap) -> Dict[str, object]:
    """Compute statistics across all playlists."""
    all_songs: List[Song] = []
    for songs in playlists.values():
        all_songs.extend(songs)

    hype = playlists.get("Hype", [])
    chill = playlists.get("Chill", [])
    mixed = playlists.get("Mixed", [])

    total = len(all_songs) # Corrected to make sure we are looking at all songs and not just the hype songs
    hype_ratio = len(hype) / total if total > 0 else 0.0

    avg_energy = 0.0
    if all_songs:
        total_energy = sum(song.get("energy", 0) for song in all_songs)
        avg_energy = total_energy / len(all_songs)

    top_artist, top_count = most_common_artist(all_songs)

    return {
        "total_songs": len(all_songs),
        "hype_count": len(hype),
        "chill_count": len(chill),
        "mixed_count": len(mixed),
        "hype_ratio": hype_ratio,
        "avg_energy": avg_energy,
        "top_artist": top_artist,
        "top_artist_count": top_count,
    }


def most_common_artist(songs: List[Song]) -> Tuple[str, int]:
    """Return the most common artist and count."""
    counts: Dict[str, int] = {}
    for song in songs:
        artist = str(song.get("artist", ""))
        if not artist:
            continue
        counts[artist] = counts.get(artist, 0) + 1

    if not counts:
        return "", 0

    items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return items[0]


def search_songs(
    songs: List[Song],
    query: str,
    field: str = "artist",
) -> List[Song]:
    """Return songs matching the query on a given field."""
    if not query:
        return songs

    q = query.lower().strip()
    filtered: List[Song] = []

    for song in songs:
        value = str(song.get(field, "")).lower()
        if q and value and q in value:
            filtered.append(song)

    return filtered


def song_to_retrieval_text(song: Song) -> str:
    """Format a song for retrieval context."""
    tags = ", ".join(song.get("tags", []))
    return (
        f"{song.get('title', '').strip()} by {song.get('artist', '').strip()} "
        f"(genre {song.get('genre', '')}, energy {song.get('energy', 0)}, tags {tags})"
    )


def compute_song_relevance(song: Song, profile: Dict[str, object], query: str) -> float:
    """Score a song based on query and profile for retrieval."""
    if not query:
        return 0.0

    query = query.lower().strip()
    tokens = set(re.findall(r"\w+", query))
    score = 0.0
    genre = song.get("genre", "")
    tags = song.get("tags", [])
    energy = song.get("energy", 0)

    if any(token in genre for token in tokens):
        score += 2.0
    if any(token in tag for tag in tags for token in tokens):
        score += 1.5
    if any(token in song.get("title", "").lower() for token in tokens):
        score += 1.0
    if any(token in song.get("artist", "").lower() for token in tokens):
        score += 0.8

    if "hype" in tokens and energy >= profile.get("hype_min_energy", 7):
        score += 1.5
    if "chill" in tokens and energy <= profile.get("chill_max_energy", 3):
        score += 1.5
    if genre == profile.get("favorite_genre", ""):
        score += 1.0

    score += max(0.0, 1.0 - abs(energy - 5) / 10.0)
    return score


def retrieve_relevant_songs(
    songs: List[Song],
    query: str,
    profile: Dict[str, object],
    top_k: int = 5,
) -> List[Song]:
    """Select the most relevant songs for an AI prompt."""
    if not songs:
        return []

    scored_songs = [
        (compute_song_relevance(song, profile, query), song)
        for song in songs
    ]
    scored_songs.sort(key=lambda item: item[0], reverse=True)
    selected = [song for score, song in scored_songs if score > 0.0]

    if len(selected) < top_k:
        selected = [song for _, song in scored_songs][:top_k]

    return selected[:top_k]


def lucky_pick(
    playlists: PlaylistMap,
    mode: str = "any",
) -> Optional[Song]:
    """Pick a song from the playlists according to mode."""
    if mode == "hype":
        songs = playlists.get("Hype", [])
    elif mode == "chill":
        songs = playlists.get("Chill", [])
    else:
        songs = playlists.get("Hype", []) + playlists.get("Chill", [])

    return random_choice_or_none(songs)


def random_choice_or_none(songs: List[Song]) -> Optional[Song]:
    """Return a random song or None."""
    import random

    return random.choice(songs)


def history_summary(history: List[Song]) -> Dict[str, int]:
    """Return a summary of moods seen in the history."""
    counts = {"Hype": 0, "Chill": 0, "Mixed": 0}
    for song in history:
        mood = song.get("mood", "Mixed")
        if mood not in counts:
            counts["Mixed"] += 1
        else:
            counts[mood] += 1
    return counts
