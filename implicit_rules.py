"""
implicit_rules.py
==================
Phase 2 (rule-based tier) — implicit relation detection between adjacent
sentences that have NO explicit connective (see README: "Phase 2 — Implicit
relations"). tokenizer.py's `extract_statement` only fires when a surface
connective like "caused" or "त्यसैले" is present; most adjacent sentence
pairs in a news article have no such marker and are invisible to Phase 1.

This module guesses a relation for those pairs from lexical cues alone, and
returns it in the same (conn_type, polarity) shape as `tokenizer.Statement`
so it can be scored by `inference.bfs_query`/`store.TripleStore` the same
way an explicit statement would — just at lower confidence.

Usage:
    from implicit_rules import classify_implicit
    result = classify_implicit(sent1, sent2)
"""
from dataclasses import dataclass, field

from tokenizer import find_connective, word_tokenize

# tokenizer.word_tokenize's fallback regex doesn't split the danda (।) off
# a sentence-final word (e.g. "भयो।" stays one token), so exact-token
# lexicon lookups below strip trailing punctuation first.
_TRAILING_PUNCT = ".,;:?!।॥"


def _clean_tokens(sentence: str) -> set[str]:
    return {t.rstrip(_TRAILING_PUNCT) for t in word_tokenize(sentence)}

# ═══════════════════════════════════════════════════════════════════════════
# 1. LEXICONS
# ═══════════════════════════════════════════════════════════════════════════
# Cause-topic nouns in sentence 1 co-occurring with effect-marker verbs in
# sentence 2 → CAUSAL, polarity +1 (mirrors tokenizer.CONNECTIVES' CAUSAL
# entries, but inferred from topic/outcome co-occurrence instead of a
# surface connective).
CAUSE_TOPICS = {
    "वर्षा", "बाढी", "भूकम्प", "आगलागी", "मूल्य", "महामारी", "हड्ताल",
}
EFFECT_MARKERS = {
    "बढ्यो", "घट्यो", "आयो", "भयो", "नष्ट", "क्षति", "मृत्यु", "बन्द",
}

# An antonym pair spanning the two sentences → ADVERSATIVE, polarity -1.
ANTONYM_PAIRS = {
    ("बढ्यो", "घट्यो"), ("घट्यो", "बढ्यो"),
    ("राम्रो", "नराम्रो"), ("नराम्रो", "राम्रो"),
    ("खुल्यो", "बन्द"), ("बन्द", "खुल्यो"),
}

# Uncertainty markers in either sentence → HEDGED, polarity 0.
HEDGE_MARKERS = {"सक्छ", "होला", "सम्भवतः", "शायद", "may", "might", "could", "allegedly"}


@dataclass
class ImplicitResult:
    conn_type: str              # CAUSAL / ADVERSATIVE / HEDGED / NONE
    polarity: int                 # +1 / -1 / 0
    confidence: float
    p: str = ""
    q: str = ""
    evidence: list[str] = field(default_factory=list)
    method: str = "implicit_rule"


def classify_implicit(sent1: str, sent2: str) -> ImplicitResult:
    """
    Classify the implicit relation between two adjacent sentences.

    Declines (conn_type="NONE") if either sentence already has an explicit
    connective — that's Phase 1's job (tokenizer.extract_statement), not
    this module's; double-counting the same relation from both tiers would
    inflate its weight in the triple store.
    """
    if find_connective(sent1) or find_connective(sent2):
        return ImplicitResult(
            conn_type="NONE", polarity=0, confidence=0.0, p=sent1, q=sent2,
            evidence=["explicit connective present, not implicit"],
        )

    tokens1 = _clean_tokens(sent1)
    tokens2 = _clean_tokens(sent2)

    # 1. Hedge markers anywhere → uncertain, regardless of other cues.
    hedges = (tokens1 | tokens2) & HEDGE_MARKERS
    if hedges:
        return ImplicitResult(
            conn_type="HEDGED", polarity=0, confidence=0.4, p=sent1, q=sent2,
            evidence=sorted(hedges),
        )

    # 2. Cause-topic in sentence 1 + effect-marker in sentence 2 → causal.
    causes = tokens1 & CAUSE_TOPICS
    effects = tokens2 & EFFECT_MARKERS
    if causes and effects:
        evidence = [f"{c} -> {e}" for c in causes for e in effects]
        return ImplicitResult(
            conn_type="CAUSAL", polarity=1, confidence=0.6, p=sent1, q=sent2,
            evidence=evidence,
        )

    # 3. Antonym pair spanning the two sentences → adversative.
    for w1 in tokens1:
        for w2 in tokens2:
            if (w1, w2) in ANTONYM_PAIRS:
                return ImplicitResult(
                    conn_type="ADVERSATIVE", polarity=-1, confidence=0.55,
                    p=sent1, q=sent2, evidence=[f"{w1} <-> {w2}"],
                )

    return ImplicitResult(conn_type="NONE", polarity=0, confidence=0.0, p=sent1, q=sent2)
