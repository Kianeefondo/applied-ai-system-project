import unittest

from playlist_ai import build_rag_prompt, is_openai_configured, local_playlist_advice
from playlist_logic import normalize_song, retrieve_relevant_songs, search_songs


class TestPlaylistAI(unittest.TestCase):
    def setUp(self):
        self.songs = [
            normalize_song({
                "title": "Calm Study",
                "artist": "DJ Focus",
                "genre": "lofi",
                "energy": 2,
                "tags": ["study", "chill"],
            }),
            normalize_song({
                "title": "Power Run",
                "artist": "The Gym",
                "genre": "rock",
                "energy": 9,
                "tags": ["hype", "workout"],
            }),
        ]
        self.profile = {
            "name": "Tester",
            "hype_min_energy": 7,
            "chill_max_energy": 3,
            "favorite_genre": "lofi",
            "include_mixed": True,
        }

    def test_search_songs_matches_substrings(self):
        matches = search_songs(self.songs, "dj focus", field="artist")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["artist"], "dj focus")

    def test_retrieve_relevant_songs_returns_top_matches(self):
        results = retrieve_relevant_songs(self.songs, "study chill", self.profile)
        self.assertTrue(results)
        self.assertEqual(results[0]["genre"], "lofi")

    def test_build_rag_prompt_includes_profile(self):
        prompt = build_rag_prompt(self.profile, {"total_songs": 2, "hype_count": 1, "chill_count": 1, "mixed_count": 0, "avg_energy": 5.5}, "Suggest a chill playlist.", self.songs)
        self.assertIn("Profile:", prompt)
        self.assertIn("total_songs", prompt)
        self.assertIn("Calm Study", prompt)

    def test_local_playlist_advice_returns_summary(self):
        advice = local_playlist_advice(self.profile, "Study with chill music.", self.songs)
        self.assertIn("Top matching songs", advice)
        self.assertIn("Calm Study", advice)

    def test_openai_configuration_is_boolean(self):
        configured = is_openai_configured()
        self.assertIsInstance(configured, bool)


if __name__ == "__main__":
    unittest.main()
