import logging
import os
from typing import Dict, List, Optional

try:
    import openai
except ImportError:
    openai = None

from playlist_logic import (
    build_playlists,
    compute_playlist_stats,
    normalize_song,
    retrieve_relevant_songs,
    song_to_retrieval_text,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.FileHandler("app.log")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def is_openai_configured() -> bool:
    """Return whether OpenAI is available from environment and package."""
    if openai is None:
        logger.warning("openai package is not installed; using local fallback.")
        return False
    return bool(os.getenv("OPENAI_API_KEY"))


def build_rag_prompt(
    profile: Dict[str, object],
    stats: Dict[str, object],
    request: str,
    songs: List[Dict[str, object]],
) -> str:
    """Construct a prompt that includes profile, stats, request, and retrieved context."""
    profile_lines = "\n".join(f"- {key}: {value}" for key, value in profile.items())
    context_lines = "\n".join(song_to_retrieval_text(song) for song in songs)

    return (
        "You are a playlist assistant. Use the available song library and user profile to answer the request. "
        "Do not invent songs that are not in the library.\n\n"
        f"User request: {request}\n\n"
        "Profile:\n"
        f"{profile_lines}\n\n"
        "Playlist stats:\n"
        f"- total_songs: {stats.get('total_songs', 0)}\n"
        f"- hype_count: {stats.get('hype_count', 0)}\n"
        f"- chill_count: {stats.get('chill_count', 0)}\n"
        f"- mixed_count: {stats.get('mixed_count', 0)}\n"
        f"- avg_energy: {stats.get('avg_energy', 0.0):.2f}\n\n"
        "Retrieved songs:\n"
        f"{context_lines}\n\n"
        "Answer concisely with a playlist recommendation and why these songs fit the request. "
        "If the request asks for a mood, explain how the selected songs match that mood."
    )


def call_openai_model(prompt: str) -> str:
    """Call the OpenAI ChatCompletion API and return assistant text."""
    if not is_openai_configured():
        raise RuntimeError("OpenAI API key is not configured.")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful playlist assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as error:
        logger.error("OpenAI call failed: %s", error, exc_info=True)
        raise


def local_playlist_advice(
    profile: Dict[str, object],
    request: str,
    songs: List[Dict[str, object]],
) -> str:
    """Produce a deterministic local fallback response when no LLM is available."""
    if not songs:
        return "No songs in the current library match your request. Add more songs or adjust your profile."

    lines = [
        "I could not call an external AI model, so I am using a local recommendation summary.",
        f"Request: {request}",
        "Top matching songs from your library:",
    ]

    for song in songs[:5]:
        lines.append(
            f"- {song.get('title')} by {song.get('artist')} "
            f"(genre {song.get('genre')}, energy {song.get('energy')})"
        )

    moods = {"Hype": 0, "Chill": 0, "Mixed": 0}
    for song in songs:
        moods[song.get("mood", "Mixed")] = moods.get(song.get("mood", "Mixed"), 0) + 1
    lines.append(
        "Recommendation note: This fallback is based on the retrieved songs and your profile mood settings."
    )
    lines.append(
        f"Mood breakdown: Hype {moods['Hype']}, Chill {moods['Chill']}, Mixed {moods['Mixed']}"
    )
    return "\n".join(lines)


def compute_confidence(
    used_openai: bool,
    retrieved_count: int,
    error: Optional[str],
) -> float:
    """Return a simple confidence score for the AI result."""
    if error:
        return 0.0

    base = 0.7 if used_openai else 0.4
    bonus = min(0.2, retrieved_count * 0.04)
    return round(min(1.0, base + bonus), 2)


def get_ai_playlist_advice(
    songs: List[Dict[str, object]],
    profile: Dict[str, object],
    request: str,
) -> Dict[str, object]:
    """Return an AI or fallback recommendation with retrieval context and confidence."""
    request_text = str(request or "").strip()
    if not request_text:
        return {
            "response": "Please enter a prompt so the AI assistant can recommend a playlist.",
            "confidence": 0.0,
            "retrieved_songs": [],
            "used_openai": False,
            "error": None,
        }

    normalized_songs = [normalize_song(song) for song in songs]
    retrieved_songs = retrieve_relevant_songs(normalized_songs, request_text, profile)
    stats = compute_playlist_stats(build_playlists(normalized_songs, profile))
    prompt = build_rag_prompt(profile, stats, request_text, retrieved_songs)
    used_openai = False
    error = None
    response_text = ""

    if is_openai_configured():
        try:
            response_text = call_openai_model(prompt)
            used_openai = True
        except Exception as exc:
            error = str(exc)
            response_text = local_playlist_advice(profile, request_text, retrieved_songs)
    else:
        response_text = local_playlist_advice(profile, request_text, retrieved_songs)

    confidence = compute_confidence(used_openai, len(retrieved_songs), error)
    logger.info(
        "AI request: %s | used_openai=%s | retrieved=%d | confidence=%.2f | error=%s",
        request_text,
        used_openai,
        len(retrieved_songs),
        confidence,
        error,
    )

    return {
        "response": response_text,
        "confidence": confidence,
        "retrieved_songs": retrieved_songs,
        "used_openai": used_openai,
        "error": error,
    }
