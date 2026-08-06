"""
tokenizer.py
============
Pipeline A — rule-based tokenization + connective detection.

Uses Nepali_nlp if installed, falls back to a built-in
Devanagari-aware tokenizer so the module always works.

Usage:
    from tokenizer import tokenize_article
    result = tokenize_article(headline, body)
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ── try to use Nepali_nlp; fall back gracefully ──────────────────────────────
try:
    from Nepali_nlp import Tokenizer as _NepTokenizer
    _NEP = _NepTokenizer()
    _USE_NEPALI_NLP = True
except ImportError:
    _USE_NEPALI_NLP = False

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONNECTIVE LEXICON
# ═══════════════════════════════════════════════════════════════════════════════
# Each entry: surface_form -> (conn_type, polarity)
#   polarity: 1 = enabling (p causes/leads to q)
#            -1 = inhibiting (p prevents q)
#             0 = hedged / uncertain

CONNECTIVES: dict[str, tuple[str, int]] = {
    # ── Nepali causal (+1) ─────────────────────────────────────────────
    "को कारण": ("CAUSAL", 1),
    "कारणले": ("CAUSAL", 1),
    "गर्दा": ("CAUSAL", 1),
    "भएकाले": ("CAUSAL", 1),
    "त्यसैले": ("CAUSAL", 1),
    "फलस्वरूप": ("CAUSAL", 1),
    "जसले": ("CAUSAL", 1),
    "परिणामस्वरूप": ("CAUSAL", 1),
    "किनभने": ("CAUSAL", 1),
    "किनकि": ("CAUSAL", 1),
    "हुनाले": ("CAUSAL", 1),
    # ── Nepali conditional (+1) ────────────────────────────────────────
    "यदि": ("CONDITIONAL", 1),
    "भने": ("CONDITIONAL", 1),
    "जब": ("CONDITIONAL", 1),
    "जब सम्म": ("CONDITIONAL", 1),
    # ── Nepali adversative (-1) ────────────────────────────────────────
    "तर": ("ADVERSATIVE", -1),
    "यद्यपि": ("ADVERSATIVE", -1),
    "भए पनि": ("ADVERSATIVE", -1),
    "तर पनि": ("ADVERSATIVE", -1),
    "बाबजुद": ("ADVERSATIVE", -1),
    "रोक्यो": ("ADVERSATIVE", -1),
    "घटायो": ("ADVERSATIVE", -1),
    # ── Nepali hedged (0) ──────────────────────────────────────────────
    "सक्छ": ("HEDGED", 0),
    "होला": ("HEDGED", 0),
    "सम्भवतः": ("HEDGED", 0),
    "अनुसार": ("HEDGED", 0),
    "भनिएको छ": ("HEDGED", 0),
    # ── English causal (+1) ────────────────────────────────────────────
    "because": ("CAUSAL", 1),
    "since": ("CAUSAL", 1),
    "caused": ("CAUSAL", 1),
    "led to": ("CAUSAL", 1),
    "resulted in": ("CAUSAL", 1),
    "therefore": ("CAUSAL", 1),
    "thus": ("CAUSAL", 1),
    "hence": ("CAUSAL", 1),
    "consequently": ("CAUSAL", 1),
    "due to": ("CAUSAL", 1),
    "owing to": ("CAUSAL", 1),
    # ── English conditional (+1) ───────────────────────────────────────
    "if": ("CONDITIONAL", 1),
    "when": ("CONDITIONAL", 1),
    "unless": ("CONDITIONAL", 1),
    # ── English adversative (-1) ───────────────────────────────────────
    "however": ("ADVERSATIVE", -1),
    "although": ("ADVERSATIVE", -1),
    "despite": ("ADVERSATIVE", -1),
    "prevented": ("ADVERSATIVE", -1),
    "stopped": ("ADVERSATIVE", -1),
    "but": ("ADVERSATIVE", -1),
    "nevertheless": ("ADVERSATIVE", -1),
    # ── English hedged (0) ─────────────────────────────────────────────
    "may": ("HEDGED", 0),
    "might": ("HEDGED", 0),
    "could": ("HEDGED", 0),
    "allegedly": ("HEDGED", 0),
    "reportedly": ("HEDGED", 0),
    "according to": ("HEDGED", 0),
}

# Sort by length descending so multi-word connectives match before sub-words
_SORTED_CONNS = sorted(CONNECTIVES.keys(), key=len, reverse=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TOKENIZERS
# ═══════════════════════════════════════════════════════════════════════════════

# Sentence boundary: Devanagari danda (।), double danda (॥), or ./?/!
_SENT_BOUNDARY = re.compile(r'(?<=[।॥.!?])\s+')


def sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences."""
    if _USE_NEPALI_NLP:
        return _NEP.sentence_tokenize(text)
    # built-in fallback
    sentences = _SENT_BOUNDARY.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def word_tokenize(sentence: str) -> list[str]:
    """Split a sentence into word tokens."""
    if _USE_NEPALI_NLP:
        return _NEP.word_tokenize(sentence)
    # built-in fallback: split on whitespace and strip punctuation
    tokens = re.findall(r'[\u0900-\u097F\w]+', sentence)
    return [t for t in tokens if t]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONNECTIVE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConnectiveMatch:
    surface: str        # the matched connective text
    conn_type: str       # CAUSAL / CONDITIONAL / ADVERSATIVE / HEDGED
    polarity: int         # +1 / -1 / 0
    char_start: int      # position in the sentence string
    char_end: int


def find_connective(sentence: str) -> Optional[ConnectiveMatch]:
    """
    Find the first (longest) connective in the sentence.
    Uses word-boundary checks so "if" does not match inside "semifinal".
    Returns None if no connective is found.
    """
    lower = sentence.lower()
    for conn in _SORTED_CONNS:
        idx = lower.find(conn)
        if idx == -1:
            continue
        end = idx + len(conn)
        # word-boundary guard
        if idx > 0 and (lower[idx - 1].isalnum() or lower[idx - 1] == "_"):
            continue
        if end < len(lower) and (lower[end].isalnum() or lower[end] == "_"):
            continue
        conn_type, polarity = CONNECTIVES[conn]
        return ConnectiveMatch(
            surface=conn,
            conn_type=conn_type,
            polarity=polarity,
            char_start=idx,
            char_end=end,
        )
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ARGUMENT SPLITTER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Statement:
    sentence: str
    p: str                  # antecedent text
    connective: str         # surface connective
    conn_type: str
    polarity: int             # +1 / -1 / 0
    q: str                    # consequent text
    confidence: float = 1.0    # 1.0 for explicit, lower for hedged
    method: str = "regex"     # extraction method: "regex", "regex_fallback", "spacy_dep"


_CLAUSE_BOUNDARY_CHARS = (',', '।')


def _nearest_boundary_after(sentence: str, start: int, limit: int) -> int:
    """Index just past the last clause-boundary char in sentence[start:limit], or start if none."""
    segment = sentence[start:limit]
    idx = max(segment.rfind(c) for c in _CLAUSE_BOUNDARY_CHARS)
    return start + idx + 1 if idx != -1 else start


def _nearest_boundary_before(sentence: str, start: int, limit: int) -> int:
    """Index of the first clause-boundary char in sentence[start:limit], or limit if none."""
    segment = sentence[start:limit]
    candidates = [segment.find(c) for c in _CLAUSE_BOUNDARY_CHARS]
    candidates = [c for c in candidates if c != -1]
    return start + min(candidates) if candidates else limit


def _split_arguments(sentence: str, match: ConnectiveMatch,
                      left_limit: int = 0, right_limit: Optional[int] = None) -> tuple[str, str]:
    """
    Pivot split: everything left of connective (back to left_limit) is ARG1 (p),
    everything right (up to right_limit) is ARG2 (q).

    left_limit/right_limit default to the full sentence, but when another
    connective is adjacent (see extract_statement), they're narrowed to the
    nearest comma/danda so one clause's text doesn't bleed into a neighbour's.

    For CAUSAL_INV connectives (due to, owing to, because of),
    the cause is on the RIGHT and effect on the LEFT — handled by
    swapping p and q in the caller.
    """
    if right_limit is None:
        right_limit = len(sentence)
    p = sentence[left_limit:match.char_start].strip().rstrip(',;:')
    q = sentence[match.char_end:right_limit].strip().lstrip(',;:')
    return p, q


_CAUSAL_INV = {"due to", "owing to", "because of", "because", "since"}


def _find_all_connectives(sentence: str) -> list[ConnectiveMatch]:
    """Find every connective in the sentence (left to right)."""
    lower = sentence.lower()
    matches = []
    for conn in _SORTED_CONNS:
        start = 0
        while True:
            idx = lower.find(conn, start)
            if idx == -1:
                break
            end = idx + len(conn)
            before_ok = idx == 0 or not (lower[idx-1].isalnum() or lower[idx-1] == "_")
            after_ok = end >= len(lower) or not (lower[end].isalnum() or lower[end] == "_")
            if before_ok and after_ok:
                matches.append(ConnectiveMatch(
                    surface=conn,
                    conn_type=CONNECTIVES[conn][0],
                    polarity=CONNECTIVES[conn][1],
                    char_start=idx,
                    char_end=end,
                ))
            start = idx + 1
    matches.sort(key=lambda m: m.char_start)
    return matches


_YADI, _BHANE = "यदि", "भने"


def _word_boundary_ok(lower: str, idx: int, end: int) -> bool:
    before_ok = idx == 0 or not (lower[idx - 1].isalnum() or lower[idx - 1] == "_")
    after_ok = end >= len(lower) or not (lower[end].isalnum() or lower[end] == "_")
    return before_ok and after_ok


def _extract_paired_yadi_bhane(sentence: str) -> Optional[Statement]:
    """
    "यदि ... भने" (if ... then) is a discontinuous conditional: यदि marks the
    start of the antecedent and भने marks its end / the start of the
    consequent. Treating them as two independent single-word connectives
    (the generic path below) leaves "यदि" stuck inside p. Handle the pair
    explicitly so p/q are clean.
    """
    lower = sentence.lower()
    yadi_idx = lower.find(_YADI)
    if yadi_idx == -1 or not _word_boundary_ok(lower, yadi_idx, yadi_idx + len(_YADI)):
        return None
    search_from = yadi_idx + len(_YADI)
    bhane_idx = lower.find(_BHANE, search_from)
    if bhane_idx == -1 or not _word_boundary_ok(lower, bhane_idx, bhane_idx + len(_BHANE)):
        return None

    p_raw = sentence[search_from:bhane_idx].strip().rstrip(',;:')
    q_raw = sentence[bhane_idx + len(_BHANE):].strip().lstrip(',;:')
    if len(p_raw.split()) < 2 or len(q_raw.split()) < 2:
        return None

    return Statement(
        sentence=sentence,
        p=p_raw,
        connective=f"{_YADI} ... {_BHANE}",
        conn_type="CONDITIONAL",
        polarity=1,
        q=q_raw,
        confidence=1.0,
    )


def extract_statement(sentence: str) -> Optional[Statement]:
    """
    Attempt to extract a single (p, connective, q) triple
    from a sentence. Returns None if no connective is found
    or if either argument span is too short to be meaningful.

    Handles the discontinuous "यदि ... भने" conditional as a paired
    marker (see _extract_paired_yadi_bhane) before falling back to the
    generic single-connective pivot search below.
    """
    paired = _extract_paired_yadi_bhane(sentence)
    if paired:
        return paired

    matches = _find_all_connectives(sentence)
    if not matches:
        return None

    for i, match in enumerate(matches):
        # bound p/q to the nearest comma/danda when another connective is
        # adjacent, so this clause's text doesn't swallow a neighbour's
        left_limit = _nearest_boundary_after(sentence, matches[i - 1].char_end, match.char_start) if i > 0 else 0
        right_limit = (_nearest_boundary_before(sentence, match.char_end, matches[i + 1].char_start)
                        if i + 1 < len(matches) else None)
        p_raw, q_raw = _split_arguments(sentence, match, left_limit, right_limit)

        # flip for inverse-causal connectives
        if match.surface in _CAUSAL_INV:
            p_raw, q_raw = q_raw, p_raw

        # reject degenerate splits (less than 2 words on either side)
        if len(p_raw.split()) < 2 or len(q_raw.split()) < 2:
            continue

        confidence = 0.6 if match.polarity == 0 else 1.0
        return Statement(
            sentence=sentence,
            p=p_raw,
            connective=match.surface,
            conn_type=match.conn_type,
            polarity=match.polarity,
            q=q_raw,
            confidence=confidence,
        )
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# 5. LANGUAGE DETECTION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def _is_devanagari(text: str) -> bool:
    """True if text contains any Devanagari characters (Nepali/Hindi)."""
    return bool(_DEVANAGARI_RE.search(text))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ARTICLE-LEVEL PIPELINE (hybrid: NLP + regex fallback)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenizedArticle:
    headline: str
    sentences: list[str]
    statements: list[Statement]
    token_count: int


def _extract_hybrid(sentence: str) -> Optional[Statement]:
    """
    Hybrid extraction for a single sentence:
      1. If the sentence is Devanagari → regex pipeline only
      2. Otherwise → try spaCy NLP extractor first
      3. Fall back to regex if NLP returns nothing
    Returns the first Statement found, or None.
    """
    # Devanagari text → regex only (no spaCy Nepali dep parser exists)
    if _is_devanagari(sentence):
        stmt = extract_statement(sentence)
        if stmt:
            stmt.method = "regex"
        return stmt

    # English text → try NLP first
    try:
        from nlp_extractor import extract_statements_nlp, is_available
        if is_available():
            nlp_stmts = extract_statements_nlp(sentence)
            if nlp_stmts:
                return nlp_stmts[0]  # best match from dep-parse
    except ImportError:
        pass

    # Fallback → regex pipeline
    stmt = extract_statement(sentence)
    if stmt:
        stmt.method = "regex_fallback"
    return stmt


def tokenize_article(headline: str, body: str) -> TokenizedArticle:
    """
    Full pipeline for one article:
      1. Sentence-tokenize headline + body
      2. Word-tokenize each sentence (for token count)
      3. Run connective detector + argument splitter on each sentence
         — uses spaCy NLP for English, regex for Nepali, with fallback
      4. Return structured result

    Headline is prepended as its own sentence so it is also
    checked for statements.
    """
    full_text = headline.strip() + "। " + body.strip()
    sentences = sentence_tokenize(full_text)

    all_tokens = []
    statements = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        all_tokens.extend(tokens)
        stmt = _extract_hybrid(sent)
        if stmt:
            statements.append(stmt)

    return TokenizedArticle(
        headline=headline,
        sentences=sentences,
        statements=statements,
        token_count=len(all_tokens),
    )
