"""
tests/test_endpoints.py
=======================
Tests for /tokenize, /assert, /query, /triples, and /analyse endpoints.

Run:
    python -m unittest discover -s tests -v
"""

import unittest
from fastapi.testclient import TestClient
from main import app, store

client = TestClient(app)


# ── helpers ──────────────────────────────────────────────────────────────────

def reset_store():
    from rdflib import Graph
    from store import TATHYA
    store.g = Graph()
    store.g.bind("tathya", TATHYA)


def assert_triple(p, polarity, q, weight=1.0):
    return client.post("/assert", json={
        "p": p, "polarity": polarity, "q": q, "weight": weight
    })


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
        self.assertGreater(len(s["p"]), 0)
        self.assertGreater(len(s["q"]), 0)

    def test_nepali_causal_statement(self):
        r = client.post("/tokenize", json={
            "headline": "इन्धन मूल्य वृद्धि",
            "body": "इन्धन मूल्य बढेको कारणले यातायात भाडा पनि बढ्यो ।",
        })
        self.assertEqual(r.status_code, 200)
        stmts = r.json()["statements"]
        self.assertGreaterEqual(len(stmts), 1)
        self.assertEqual(stmts[0]["conn_type"], "CAUSAL")
        self.assertEqual(stmts[0]["polarity"], 1)

    def test_adversative_gives_negative_polarity(self):
        r = client.post("/tokenize", json={
            "headline": "Policy impact",
            "body": "The central bank's intervention prevented inflation from rising.",
        })
        self.assertEqual(r.status_code, 200)
        stmts = r.json()["statements"]
        self.assertGreaterEqual(len(stmts), 1)
        self.assertEqual(stmts[0]["polarity"], -1)
        self.assertEqual(stmts[0]["conn_type"], "ADVERSATIVE")

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

    def test_multiple_statements_extracted(self):
        r = client.post("/tokenize", json={
            "headline": "Economic overview",
            "body": (
                "The fuel price hike caused transport costs to rise. "
                "However, the central bank's intervention prevented further damage. "
                "The drought may reduce grain output."
            ),
        })
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(len(r.json()["statements"]), 2)

    def test_empty_body_returns_200(self):
        r = client.post("/tokenize", json={"headline": "Breaking news", "body": ""})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["statements"], [])

    def test_nepali_conditional(self):
        r = client.post("/tokenize", json={
            "headline": "अर्थ समाचार",
            "body": "यदि ब्याज दर बढ्यो भने लगानी घट्छ ।",
        })
        self.assertEqual(r.status_code, 200)
        stmts = r.json()["statements"]
        self.assertGreaterEqual(len(stmts), 1)
        self.assertEqual(stmts[0]["conn_type"], "CONDITIONAL")
        self.assertEqual(stmts[0]["polarity"], 1)


# ═══════════════════════════════════════════════════════════════════════════════
# /assert
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssert(unittest.TestCase):

    def setUp(self):
        reset_store()

    def test_create_positive_triple(self):
        r = assert_triple("fuel_price_hike", 1, "transport_cost_rise")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "created")
        self.assertEqual(data["polarity"], 1)
        self.assertAlmostEqual(data["weight"], 1.0)

    def test_create_negative_triple(self):
        r = assert_triple("central_bank_intervention", -1, "inflation_rise")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "created")
        self.assertEqual(r.json()["polarity"], -1)

    def test_duplicate_assertion_accumulates_weight(self):
        payload = {"p": "drought", "polarity": 1, "q": "grain_shortage"}
        client.post("/assert", json=payload)
        r = client.post("/assert", json=payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "updated")
        self.assertAlmostEqual(r.json()["weight"], 2.0)

    def test_custom_weight(self):
        r = assert_triple("rupee_fall", 1, "import_cost_rise", weight=2.5)
        self.assertEqual(r.status_code, 200)
        self.assertAlmostEqual(r.json()["weight"], 2.5)

    def test_invalid_polarity_rejected(self):
        r = client.post("/assert", json={"p": "x", "polarity": 0, "q": "y"})
        self.assertEqual(r.status_code, 422)

    def test_empty_p_rejected(self):
        r = client.post("/assert", json={"p": "", "polarity": 1, "q": "something"})
        self.assertEqual(r.status_code, 422)

    def test_empty_q_rejected(self):
        r = client.post("/assert", json={"p": "something", "polarity": 1, "q": "   "})
        self.assertEqual(r.status_code, 422)

    def test_triples_endpoint_reflects_assertion(self):
        assert_triple("a", 1, "b")
        r = client.get("/triples")
        self.assertEqual(r.status_code, 200)
        triples = r.json()["triples"]
        self.assertTrue(any(t["p"] == "a" and t["q"] == "b" for t in triples))


# ═══════════════════════════════════════════════════════════════════════════════
# /query
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuery(unittest.TestCase):

    def setUp(self):
        reset_store()
        assert_triple("fuel_price_hike",     1,  "transport_cost_rise")
        assert_triple("transport_cost_rise", 1,  "inflation_rise")
        assert_triple("central_bank_act",   -1,  "inflation_rise")

    def test_direct_path_found(self):
        r = client.post("/query", json={
            "p": "fuel_price_hike", "q": "transport_cost_rise",
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["known"])
        self.assertAlmostEqual(data["bias_score"], 1.0)
        self.assertEqual(len(data["paths"]), 1)
        self.assertEqual(data["paths"][0]["polarity"], 1)

    def test_transitive_path_found(self):
        r = client.post("/query", json={
            "p": "fuel_price_hike", "q": "inflation_rise",
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["known"])
        self.assertAlmostEqual(data["bias_score"], 1.0)
        self.assertEqual(
            data["paths"][0]["chain"],
            ["fuel_price_hike", "transport_cost_rise", "inflation_rise"]
        )

    def test_inhibiting_path(self):
        r = client.post("/query", json={
            "p": "central_bank_act", "q": "inflation_rise",
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["known"])
        self.assertAlmostEqual(data["bias_score"], -1.0)
        self.assertEqual(data["paths"][0]["polarity"], -1)

    def test_unknown_returns_not_known(self):
        r = client.post("/query", json={"p": "drought", "q": "food_price_rise"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["known"])
        self.assertIsNone(data["bias_score"])
        self.assertEqual(data["paths"], [])

    def test_verdict_agrees(self):
        r = client.post("/query", json={
            "p": "fuel_price_hike", "q": "transport_cost_rise",
            "expected_polarity": 1,
        })
        self.assertEqual(r.json()["verdict"], "agrees")

    def test_verdict_contradicts(self):
        r = client.post("/query", json={
            "p": "central_bank_act", "q": "inflation_rise",
            "expected_polarity": 1,
        })
        self.assertEqual(r.json()["verdict"], "contradicts")

    def test_verdict_unknown_when_no_path(self):
        r = client.post("/query", json={
            "p": "drought", "q": "export_decline", "expected_polarity": 1,
        })
        self.assertEqual(r.json()["verdict"], "unknown")

    def test_self_query_returns_known(self):
        r = client.post("/query", json={
            "p": "fuel_price_hike", "q": "fuel_price_hike",
        })
        self.assertTrue(r.json()["known"])

    def test_empty_p_rejected(self):
        r = client.post("/query", json={"p": "", "q": "something"})
        self.assertEqual(r.status_code, 422)

    def test_double_negative_gives_positive_polarity(self):
        reset_store()
        assert_triple("a", -1, "b")
        assert_triple("b", -1, "c")
        r = client.post("/query", json={"p": "a", "q": "c"})
        data = r.json()
        self.assertTrue(data["known"])
        self.assertEqual(data["paths"][0]["polarity"], 1)
        self.assertAlmostEqual(data["bias_score"], 1.0)

    def test_multiple_paths_averaged_in_bias(self):
        reset_store()
        assert_triple("x",  1, "z")
        assert_triple("x", -1, "y")
        assert_triple("y",  1, "z")
        r = client.post("/query", json={"p": "x", "q": "z"})
        data = r.json()
        self.assertTrue(data["known"])
        self.assertEqual(len(data["paths"]), 2)
        self.assertLess(abs(data["bias_score"]), 1.0)


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
        assert_triple("a", 1, "b")
        assert_triple("b", 1, "c")
        r = client.get("/triples")
        self.assertEqual(r.json()["count"], 2)


# ═══════════════════════════════════════════════════════════════════════════════
# /analyse
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyse(unittest.TestCase):

    def setUp(self):
        reset_store()

    def test_returns_expected_top_level_keys(self):
        r = client.post("/analyse", json={
            "headline": "Test",
            "body": "The fuel price hike caused transport costs to rise.",
        })
        self.assertEqual(r.status_code, 200)
        for k in ("headline", "source", "token_count", "sentence_count",
                  "statement_count", "known_count", "new_count",
                  "article_bias", "statements"):
            self.assertIn(k, r.json())

    def test_empty_body_rejected(self):
        r = client.post("/analyse", json={"headline": "x", "body": "   "})
        self.assertEqual(r.status_code, 422)

    def test_source_echoed_in_response(self):
        r = client.post("/analyse", json={
            "headline": "h",
            "body": "Drought caused food shortage.",
            "source": "setopati",
        })
        self.assertEqual(r.json()["source"], "setopati")

    def test_unknown_statement_marked_new(self):
        r = client.post("/analyse", json={
            "headline": "",
            "body": "The drought caused a food shortage across the region.",
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertGreaterEqual(data["new_count"], 1)
        stmt = data["statements"][0]
        self.assertEqual(stmt["status"], "new")
        self.assertEqual(stmt["verdict"], "no_prior_data")
        self.assertIsNone(stmt["bias_score"])
        self.assertEqual(stmt["paths"], [])

    def test_new_fact_ingested_into_store(self):
        body = "The drought caused a food shortage across the region."
        r = client.post("/analyse", json={"headline": "", "body": body})
        stmt = r.json()["statements"][0]
        p_key = stmt["p"].strip().lower().replace(" ", "_")
        q_key = stmt["q"].strip().lower().replace(" ", "_")
        qr = client.post("/query", json={"p": p_key, "q": q_key})
        self.assertTrue(qr.json()["known"])

    def test_hedged_statement_not_ingested(self):
        body = "The drought may reduce grain output in the Terai region."
        r = client.post("/analyse", json={"headline": "", "body": body})
        stmts = r.json()["statements"]
        hedged = [s for s in stmts if s["conn_type"] == "HEDGED"]
        self.assertGreaterEqual(len(hedged), 1)
        p_key = hedged[0]["p"].strip().lower().replace(" ", "_")
        q_key = hedged[0]["q"].strip().lower().replace(" ", "_")
        qr = client.post("/query", json={"p": p_key, "q": q_key})
        self.assertFalse(qr.json()["known"])

    def test_new_fact_seeded_with_lower_weight(self):
        client.post("/analyse", json={
            "headline": "",
            "body": "The floods caused infrastructure damage in the valley.",
        })
        triples = client.get("/triples").json()["triples"]
        new_triples = [t for t in triples if abs(t["weight"] - 0.5) < 0.01]
        self.assertGreaterEqual(len(new_triples), 1)

    def test_known_statement_marked_supported(self):
        assert_triple(
            "the_fuel_price_hike", 1,
            "transport_costs_to_rise_significantly.", weight=2.0
        )
        r = client.post("/analyse", json={
            "headline": "Fuel prices rise",
            "body": "The fuel price hike caused transport costs to rise significantly.",
        })
        data = r.json()
        self.assertEqual(data["known_count"], 1)
        stmt = data["statements"][0]
        self.assertEqual(stmt["status"], "known")
        self.assertEqual(stmt["verdict"], "supported")
        self.assertAlmostEqual(stmt["bias_score"], 2.0)
        self.assertGreaterEqual(len(stmt["paths"]), 1)

    def test_known_contradicted_statement(self):
        assert_triple("central_bank_action", 1, "inflation_rise.")
        r = client.post("/analyse", json={
            "headline": "",
            "body": "Central bank action prevented inflation rise.",
        })
        stmts = r.json()["statements"]
        self.assertGreaterEqual(len(stmts), 1)
        self.assertEqual(stmts[0]["status"], "known")
        self.assertEqual(stmts[0]["verdict"], "contradicted")

    def test_article_bias_is_float_when_known(self):
        assert_triple(
            "the_fuel_price_hike", 1,
            "transport_costs_to_rise_significantly.", weight=1.0
        )
        r = client.post("/analyse", json={
            "headline": "Fuel prices rise",
            "body": "The fuel price hike caused transport costs to rise significantly.",
        })
        bias = r.json()["article_bias"]
        self.assertIsNotNone(bias)
        self.assertIsInstance(bias, float)

    def test_article_bias_none_when_all_new(self):
        r = client.post("/analyse", json={
            "headline": "",
            "body": "Rainfall caused flooding in the southern districts.",
        })
        self.assertIsNone(r.json()["article_bias"])

    def test_known_and_new_counts_sum_to_total(self):
        assert_triple(
            "the_fuel_price_hike", 1,
            "transport_costs_to_rise_significantly.", weight=1.0
        )
        r = client.post("/analyse", json={
            "headline": "Mixed article",
            "body": (
                "The fuel price hike caused transport costs to rise significantly. "
                "The drought may reduce grain output in the Terai region."
            ),
        })
        data = r.json()
        self.assertEqual(
            data["known_count"] + data["new_count"],
            data["statement_count"]
        )

    def test_no_statements_gives_zero_counts(self):
        r = client.post("/analyse", json={
            "headline": "",
            "body": "Nepal beat Bangladesh in the football semifinal.",
        })
        data = r.json()
        self.assertEqual(data["statement_count"], 0)
        self.assertEqual(data["known_count"], 0)
        self.assertEqual(data["new_count"], 0)
        self.assertIsNone(data["article_bias"])


if __name__ == "__main__":
    unittest.main()
