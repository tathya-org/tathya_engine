"""
Tests for /tokenize, /assert, and /query endpoints using unittest.

Run:
    python -m unittest discover -s tests -v
"""

import unittest
from fastapi.testclient import TestClient
from main import app, store

client = TestClient(app)


# ── helpers ──────────────────────────────────────────────────────────────────

def reset_store():
    """Wipe the in-memory graph between tests."""
    from rdflib import Graph
    from store import TATHYA
    store.g = Graph()
    store.g.bind("tathya", TATHYA)


# ═══════════════════════════════════════════════════════════════════════════════
# /tokenize
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenize(unittest.TestCase):

    def test_english_causal_statement(self):
        r = client.post("/tokenize", json={
            "headline": "Fuel prices rise",
            "body": "The fuel price hike caused transport costs to rise significantly.",
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertGreater(data["token_count"], 0)
        self.assertGreaterEqual(len(data["sentences"]), 1)

        stmts = data["statements"]
        self.assertGreaterEqual(len(stmts), 1)
        s = stmts[0]
        self.assertEqual(s["polarity"], 1)
        self.assertEqual(s["conn_type"], "CAUSAL")
        self.assertIn("caused", s["connective"])
        self.assertTrue(len(s["p"]) > 0)
        self.assertTrue(len(s["q"]) > 0)

    def test_hedged_gives_zero_polarity(self):
        r = client.post("/tokenize", json={
            "headline": "Economic forecast",
            "body": "The drought may reduce grain output across the Terai region.",
        })
        self.assertEqual(r.status_code, 200)
        stmts = r.json()["statements"]
        self.assertGreaterEqual(len(stmts), 1)
        self.assertEqual(stmts[0]["polarity"], 0)
        self.assertEqual(stmts[0]["conn_type"], "HEDGED")
        self.assertAlmostEqual(stmts[0]["confidence"], 0.6, places=2)

    def test_no_connective_gives_empty_statements(self):
        r = client.post("/tokenize", json={
            "headline": "Sports update",
            "body": "Nepal beat Bangladesh in the football semifinal on Wednesday.",
        })
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["statements"], [])


# ═══════════════════════════════════════════════════════════════════════════════
# /assert
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssert(unittest.TestCase):

    def setUp(self):
        reset_store()

    def test_create_positive_triple(self):
        r = client.post("/assert", json={
            "p": "fuel_price_hike",
            "polarity": 1,
            "q": "transport_cost_rise",
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "created")
        self.assertEqual(data["polarity"], 1)
        self.assertAlmostEqual(data["weight"], 1.0)

    def test_duplicate_assertion_accumulates_weight(self):
        payload = {"p": "drought", "polarity": 1, "q": "grain_shortage"}
        client.post("/assert", json=payload)
        r = client.post("/assert", json=payload)

        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "updated")
        self.assertAlmostEqual(data["weight"], 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# /query
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuery(unittest.TestCase):

    def setUp(self):
        reset_store()

        triples = [
            ("fuel_price_hike",     1,  "transport_cost_rise"),
            ("transport_cost_rise", 1,  "inflation_rise"),
            ("central_bank_act",   -1,  "inflation_rise"),
        ]

        for p, pol, q in triples:
            client.post("/assert", json={"p": p, "polarity": pol, "q": q})

    def test_direct_path_found(self):
        r = client.post("/query", json={
            "p": "fuel_price_hike",
            "q": "transport_cost_rise",
        })

        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["known"])
        self.assertAlmostEqual(data["bias_score"], 1.0)
        self.assertEqual(len(data["paths"]), 1)
        self.assertEqual(data["paths"][0]["polarity"], 1)

    def test_unknown_returns_not_known(self):
        r = client.post("/query", json={
            "p": "drought",
            "q": "food_price_rise",
        })

        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["known"])
        self.assertIsNone(data["bias_score"])
        self.assertEqual(data["paths"], [])


# ═══════════════════════════════════════════════════════════════════════════════
# /triples
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriples(unittest.TestCase):

    def setUp(self):
        reset_store()

    def test_empty_store(self):
        r = client.get("/triples")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 0)

    def test_count_matches_assertions(self):
        client.post("/assert", json={"p": "a", "polarity": 1, "q": "b"})
        client.post("/assert", json={"p": "b", "polarity": 1, "q": "c"})
        r = client.get("/triples")
        self.assertEqual(r.json()["count"], 2)


if __name__ == "__main__":
    unittest.main()
