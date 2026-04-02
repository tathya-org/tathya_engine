"""
tests/test_endpoints.py
=======================
Tests for /tokenize, /assert, and /query endpoints.

Run from the project root:
    pytest tests/ -v
"""

import pytest
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

class TestTokenize:

    def test_english_causal_statement(self):
        r = client.post("/tokenize", json={
            "headline": "Fuel prices rise",
            "body": "The fuel price hike caused transport costs to rise significantly.",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["token_count"] > 0
        assert len(data["sentences"]) >= 1

        stmts = data["statements"]
        assert len(stmts) >= 1
        s = stmts[0]
        assert s["polarity"] == 1
        assert s["conn_type"] == "CAUSAL"
        assert "caused" in s["connective"]
        assert len(s["p"]) > 0
        assert len(s["q"]) > 0

    def test_nepali_causal_statement(self):
        r = client.post("/tokenize", json={
            "headline": "इन्धन मूल्य वृद्धि",
            "body": "इन्धन मूल्य बढेको कारणले यातायात भाडा पनि बढ्यो ।",
        })
        assert r.status_code == 200
        data = r.json()
        stmts = data["statements"]
        assert len(stmts) >= 1
        assert stmts[0]["conn_type"] == "CAUSAL"
        assert stmts[0]["polarity"] == 1

    def test_adversative_gives_negative_polarity(self):
        r = client.post("/tokenize", json={
            "headline": "Policy impact",
            "body": "The central bank's intervention prevented inflation from rising.",
        })
        assert r.status_code == 200
        stmts = r.json()["statements"]
        assert len(stmts) >= 1
        assert stmts[0]["polarity"] == -1
        assert stmts[0]["conn_type"] == "ADVERSATIVE"

    def test_hedged_gives_zero_polarity(self):
        r = client.post("/tokenize", json={
            "headline": "Economic forecast",
            "body": "The drought may reduce grain output across the Terai region.",
        })
        assert r.status_code == 200
        stmts = r.json()["statements"]
        assert len(stmts) >= 1
        assert stmts[0]["polarity"] == 0
        assert stmts[0]["conn_type"] == "HEDGED"
        assert stmts[0]["confidence"] == pytest.approx(0.6)

    def test_no_connective_gives_empty_statements(self):
        r = client.post("/tokenize", json={
            "headline": "Sports update",
            "body": "Nepal beat Bangladesh in the football semifinal on Wednesday.",
        })
        assert r.status_code == 200
        assert r.json()["statements"] == []

    def test_multiple_statements_extracted(self):
        r = client.post("/tokenize", json={
            "headline": "Economic overview",
            "body": (
                "The fuel price hike caused transport costs to rise. "
                "However, the central bank's intervention prevented further damage. "
                "The drought may reduce grain output."
            ),
        })
        assert r.status_code == 200
        stmts = r.json()["statements"]
        assert len(stmts) >= 2

    def test_empty_body_returns_200(self):
        r = client.post("/tokenize", json={
            "headline": "Breaking news",
            "body": "",
        })
        assert r.status_code == 200
        assert r.json()["statements"] == []

    def test_nepali_conditional(self):
        r = client.post("/tokenize", json={
            "headline": "अर्थ समाचार",
            "body": "यदि ब्याज दर बढ्यो भने लगानी घट्छ ।",
        })
        assert r.status_code == 200
        stmts = r.json()["statements"]
        assert len(stmts) >= 1
        assert stmts[0]["conn_type"] == "CONDITIONAL"
        assert stmts[0]["polarity"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# /assert
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssert:

    def setup_method(self):
        reset_store()

    def test_create_positive_triple(self):
        r = client.post("/assert", json={
            "p": "fuel_price_hike",
            "polarity": 1,
            "q": "transport_cost_rise",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "created"
        assert data["polarity"] == 1
        assert data["weight"] == 1.0

    def test_create_negative_triple(self):
        r = client.post("/assert", json={
            "p": "central_bank_intervention",
            "polarity": -1,
            "q": "inflation_rise",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "created"
        assert r.json()["polarity"] == -1

    def test_duplicate_assertion_accumulates_weight(self):
        payload = {"p": "drought", "polarity": 1, "q": "grain_shortage"}
        client.post("/assert", json=payload)
        r = client.post("/assert", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "updated"
        assert data["weight"] == pytest.approx(2.0)

    def test_custom_weight(self):
        r = client.post("/assert", json={
            "p": "rupee_fall",
            "polarity": 1,
            "q": "import_cost_rise",
            "weight": 2.5,
        })
        assert r.status_code == 200
        assert r.json()["weight"] == pytest.approx(2.5)

    def test_invalid_polarity_rejected(self):
        r = client.post("/assert", json={
            "p": "x", "polarity": 0, "q": "y",
        })
        assert r.status_code == 422

    def test_empty_p_rejected(self):
        r = client.post("/assert", json={
            "p": "", "polarity": 1, "q": "something",
        })
        assert r.status_code == 422

    def test_empty_q_rejected(self):
        r = client.post("/assert", json={
            "p": "something", "polarity": 1, "q": "   ",
        })
        assert r.status_code == 422

    def test_triples_endpoint_reflects_assertion(self):
        client.post("/assert", json={"p": "a", "polarity": 1, "q": "b"})
        r = client.get("/triples")
        assert r.status_code == 200
        triples = r.json()["triples"]
        assert any(t["p"] == "a" and t["q"] == "b" for t in triples)


# ═══════════════════════════════════════════════════════════════════════════════
# /query
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuery:

    def setup_method(self):
        reset_store()
        # seed a small knowledge graph for all query tests
        #
        #  fuel_price_hike  -+1-> transport_cost_rise -+1-> inflation_rise
        #  central_bank_act --1-> inflation_rise
        #
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
        assert r.status_code == 200
        data = r.json()
        assert data["known"] is True
        assert data["bias_score"] == pytest.approx(1.0)
        assert len(data["paths"]) == 1
        assert data["paths"][0]["polarity"] == 1

    def test_transitive_path_found(self):
        r = client.post("/query", json={
            "p": "fuel_price_hike",
            "q": "inflation_rise",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["known"] is True
        assert data["bias_score"] == pytest.approx(1.0)
        chain = data["paths"][0]["chain"]
        assert chain == ["fuel_price_hike", "transport_cost_rise", "inflation_rise"]

    def test_inhibiting_path(self):
        r = client.post("/query", json={
            "p": "central_bank_act",
            "q": "inflation_rise",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["known"] is True
        assert data["bias_score"] == pytest.approx(-1.0)
        assert data["paths"][0]["polarity"] == -1

    def test_unknown_returns_not_known(self):
        r = client.post("/query", json={
            "p": "drought",
            "q": "food_price_rise",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["known"] is False
        assert data["bias_score"] is None
        assert data["paths"] == []

    def test_verdict_agrees_when_matching_polarity(self):
        r = client.post("/query", json={
            "p": "fuel_price_hike",
            "q": "transport_cost_rise",
            "expected_polarity": 1,
        })
        assert r.json()["verdict"] == "agrees"

    def test_verdict_contradicts_when_opposing_polarity(self):
        r = client.post("/query", json={
            "p": "central_bank_act",
            "q": "inflation_rise",
            "expected_polarity": 1,   # KB says -1
        })
        assert r.json()["verdict"] == "contradicts"

    def test_verdict_unknown_when_no_path(self):
        r = client.post("/query", json={
            "p": "drought",
            "q": "export_decline",
            "expected_polarity": 1,
        })
        assert r.json()["verdict"] == "unknown"

    def test_self_query_returns_known(self):
        r = client.post("/query", json={
            "p": "fuel_price_hike",
            "q": "fuel_price_hike",
        })
        assert r.status_code == 200
        assert r.json()["known"] is True

    def test_empty_p_rejected(self):
        r = client.post("/query", json={"p": "", "q": "something"})
        assert r.status_code == 422

    def test_polarity_product_through_double_negative(self):
        """
        A -(-1)-> B -(-1)-> C  should give cumulative polarity +1
        (two negatives cancel).
        """
        reset_store()
        client.post("/assert", json={"p": "a", "polarity": -1, "q": "b"})
        client.post("/assert", json={"p": "b", "polarity": -1, "q": "c"})
        r = client.post("/query", json={"p": "a", "q": "c"})
        assert r.status_code == 200
        data = r.json()
        assert data["known"] is True
        assert data["paths"][0]["polarity"] == 1
        assert data["bias_score"] == pytest.approx(1.0)

    def test_multiple_paths_averaged_in_bias(self):
        """
        Two paths to the same target with opposite polarities
        should average to a bias near 0.
        """
        reset_store()
        client.post("/assert", json={"p": "x", "polarity":  1, "q": "z"})
        client.post("/assert", json={"p": "x", "polarity": -1, "q": "y"})
        client.post("/assert", json={"p": "y", "polarity":  1, "q": "z"})
        r = client.post("/query", json={"p": "x", "q": "z"})
        data = r.json()
        assert data["known"] is True
        assert len(data["paths"]) == 2
        assert abs(data["bias_score"]) < 1.0   # averaged, not extreme


# ═══════════════════════════════════════════════════════════════════════════════
# /triples  (smoke tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriples:

    def setup_method(self):
        reset_store()

    def test_empty_store(self):
        r = client.get("/triples")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_count_matches_assertions(self):
        client.post("/assert", json={"p": "a", "polarity": 1, "q": "b"})
        client.post("/assert", json={"p": "b", "polarity": 1, "q": "c"})
        r = client.get("/triples")
        assert r.json()["count"] == 2
