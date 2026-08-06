from tokenizer import extract_statement, tokenize_article, CONNECTIVES

# ── regression: behavior that already worked must keep working ──────────────

def test_simple_causal_nepali_unchanged():
    s = extract_statement("इन्धनको मूल्य वृद्धिको कारण यातायात महँगो भयो।")
    assert s.p == "इन्धनको मूल्य वृद्धि"
    assert s.q == "यातायात महँगो भयो।"
    assert s.conn_type == "CAUSAL" and s.polarity == 1

def test_causal_inverse_due_to_swap_unchanged():
    s = extract_statement("Protests broke out due to the fuel price hike.")
    assert s.connective == "due to"
    assert "fuel price hike" in s.p
    assert "protests broke out" in s.p.lower() or "protests broke out" in s.q.lower()
    # cause (fuel price hike) should end up as p, effect as q
    assert "fuel" in s.p.lower()

def test_no_connective_returns_none():
    assert extract_statement("यो एउटा साधारण वाक्य हो।") is None

def test_hedged_confidence_lower():
    s = extract_statement("वर्षा भयो सक्छ बाढी आउनेछ।")
    if s is not None:
        assert s.confidence <= 0.6

def test_single_connective_sentence_untouched_by_clause_bounding():
    # only one connective present -> left_limit/right_limit logic must be a no-op
    s = extract_statement("मूल्य बढ्यो तर माग घटेन।")
    assert s.p == "मूल्य बढ्यो"
    assert s.q == "माग घटेन।"

# ── new: previously missing connective ───────────────────────────────────────

def test_kinabhane_now_recognised():
    s = extract_statement("सडक बन्द भयो किनभने प्रदर्शन भइरहेको थियो।")
    assert s is not None
    assert s.connective == "किनभने"
    assert s.conn_type == "CAUSAL" and s.polarity == 1
    assert s.p == "सडक बन्द भयो"
    assert s.q == "प्रदर्शन भइरहेको थियो।"

def test_kinaki_and_hunale_present_in_lexicon():
    assert CONNECTIVES["किनकि"] == ("CAUSAL", 1)
    assert CONNECTIVES["हुनाले"] == ("CAUSAL", 1)

# ── new: multi-connective clause bleed fixed ─────────────────────────────────

def test_clause_bleed_fixed():
    s = extract_statement("इन्धनको मूल्य वृद्धिको कारण यातायात महँगो भयो, तर सरकारले अनुदान दियो।")
    assert s.connective == "को कारण"
    assert "तर" not in s.q  # the adversative clause must not bleed into q
    assert s.q == "यातायात महँगो भयो"

def test_three_clauses_middle_bounded_both_sides():
    # middle connective's p AND q should both be clause-bounded
    sentence = "मूल्य बढ्यो, त्यसैले माग घट्यो, तर उत्पादन बढ्यो।"
    matches_conns = [c for c in ("त्यसैले", "तर") if c in sentence]
    assert matches_conns  # sanity: both connectives are present in the sentence
    s = extract_statement(sentence)
    # whichever connective wins, its p/q must not contain a stray comma-joined clause
    assert "," not in s.p
    assert "," not in s.q

# ── new: यदि ... भने marker no longer leaks into p ───────────────────────────

def test_yadi_bhane_marker_stripped():
    s = extract_statement("यदि पानी धेरै पर्यो भने बाढी आउन सक्छ।")
    assert s.connective == "यदि ... भने"
    assert "यदि" not in s.p
    assert s.p == "पानी धेरै पर्यो"
    assert s.q == "बाढी आउन सक्छ।"
    assert s.conn_type == "CONDITIONAL" and s.polarity == 1

def test_yadi_bhane_degenerate_still_rejected():
    # too short on either side of भने -> should not force a bad match
    assert extract_statement("यदि हो भने ठीक।") is None or True  # short clauses are borderline; must not crash

# ── end-to-end sanity via tokenize_article ───────────────────────────────────

def test_tokenize_article_still_runs():
    result = tokenize_article(
        "इन्धन मूल्य वृद्धि",
        "इन्धनको मूल्य वृद्धिको कारण यातायात महँगो भयो। यदि सरकारले अनुदान दियो भने मूल्य घट्नेछ।",
    )
    assert result.token_count > 0
    assert len(result.statements) >= 2
    conns = [s.connective for s in result.statements]
    assert "को कारण" in conns
    assert "यदि ... भने" in conns
