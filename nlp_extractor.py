"""
nlp_extractor.py
================
Pipeline B — spaCy dependency-parse-based connective detection
and argument extraction.

Uses `en_core_web_sm` to parse English text and walks the
dependency tree to split (p, connective, q) triples properly,
avoiding the naive left/right string split of the regex pipeline.

Falls back gracefully if spaCy or the model is unavailable.

Usage:
    from nlp_extractor import extract_statements_nlp, is_available
    if is_available():
        stmts = extract_statements_nlp(text)
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ── re-use the shared Statement dataclass and lexicon from tokenizer ─────────
from tokenizer import CONNECTIVES, Statement, ConnectiveMatch

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LAZY-LOAD spaCy
# ═══════════════════════════════════════════════════════════════════════════════

_nlp = None
_AVAILABLE: Optional[bool] = None   # None = not checked yet


def _load_nlp():
    """Load spaCy model once; cache the result."""
    global _nlp, _AVAILABLE
    if _AVAILABLE is not None:
        return _nlp

    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        _AVAILABLE = True
    except Exception:
        _nlp = None
        _AVAILABLE = False

    return _nlp


def is_available() -> bool:
    """True if spaCy + en_core_web_sm are usable."""
    _load_nlp()
    return bool(_AVAILABLE)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CONNECTIVE TOKEN MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

# Build a set of English-only connective surface forms (lowercased)
# Multi-word connectives are stored with their word count so we can
# match token spans of the right length.

_EN_CONNECTIVES: dict[str, tuple[str, int]] = {
    k: v for k, v in CONNECTIVES.items()
    if all(ord(c) < 0x0900 or ord(c) > 0x097F for c in k)
}

# Group by word count for span matching
_CONN_BY_LEN: dict[int, dict[str, tuple[str, int]]] = {}
for _surf, _meta in _EN_CONNECTIVES.items():
    _wc = len(_surf.split())
    _CONN_BY_LEN.setdefault(_wc, {})[_surf] = _meta


def _find_connective_spans(doc) -> list[tuple]:
    """
    Find all connective matches in a spaCy Doc.

    Returns list of (start_token_idx, end_token_idx, surface, conn_type, polarity).
    Sorted by start index; longer matches take priority.
    """
    matches = []
    used = set()   # token indices already consumed

    # Try longest-first (multi-word before single-word)
    for wc in sorted(_CONN_BY_LEN.keys(), reverse=True):
        conns = _CONN_BY_LEN[wc]
        for i in range(len(doc) - wc + 1):
            if any(j in used for j in range(i, i + wc)):
                continue
            span_text = doc[i:i + wc].text.lower()
            if span_text in conns:
                conn_type, polarity = conns[span_text]
                matches.append((i, i + wc, span_text, conn_type, polarity))
                for j in range(i, i + wc):
                    used.add(j)

    matches.sort(key=lambda m: m[0])
    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DEPENDENCY-TREE ARGUMENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

# Connectives whose cause appears AFTER the connective (on the right)
# and whose effect appears BEFORE (on the left).
_CAUSAL_INV = {"due to", "owing to", "because of", "because", "since"}


@dataclass
class NLPStatement(Statement):
    """Statement with additional NLP metadata."""
    method:   str = "spacy_dep"    # extraction method
    dep_rel:  str = ""             # dependency relation that anchored the connective


def _subtree_span(token) -> tuple[int, int]:
    """Return (start, end) token indices of a token's full subtree."""
    subtree = sorted(token.subtree, key=lambda t: t.i)
    return subtree[0].i, subtree[-1].i + 1


def _span_text(doc, start: int, end: int) -> str:
    """Clean text from a token span, stripping leading/trailing punct."""
    tokens = [t for t in doc[start:end] if not t.is_punct and not t.is_space]
    if not tokens:
        return doc[start:end].text.strip()
    real_start = tokens[0].i
    real_end = tokens[-1].i + 1
    return doc[real_start:real_end].text.strip()


def _extract_via_mark(doc, conn_start: int, conn_end: int,
                      surface: str, conn_type: str, polarity: int) -> Optional[NLPStatement]:
    """
    Extract arguments when the connective is a `mark` dependent.

    Pattern: "Because X happened, Y followed."
             mark→advcl→ROOT

    The `mark` token's head (the advcl verb) owns the subordinate clause.
    The advcl verb's head (ROOT) owns the main clause.
    """
    conn_token = doc[conn_start]

    if conn_token.dep_ != "mark":
        return None

    sub_verb = conn_token.head          # the advcl verb
    main_verb = sub_verb.head           # the ROOT / main clause verb

    if sub_verb == main_verb:
        return None   # degenerate

    # subordinate clause = sub_verb's subtree (includes the connective)
    sub_start, sub_end = _subtree_span(sub_verb)
    # main clause = everything NOT in subordinate clause
    main_tokens = [t for t in doc if (t.i < sub_start or t.i >= sub_end) and not t.is_punct and not t.is_space]

    if not main_tokens:
        return None

    main_start = main_tokens[0].i
    main_end = main_tokens[-1].i + 1

    # The subordinate clause minus the connective itself is one argument
    sub_text_tokens = [t for t in doc[sub_start:sub_end]
                       if t.i < conn_start or t.i >= conn_end]
    sub_text = " ".join(t.text for t in sub_text_tokens if not t.is_punct and not t.is_space).strip()
    main_text = _span_text(doc, main_start, main_end)

    if not sub_text or not main_text:
        return None

    # Assign p/q: for causal-inverse connectives, the subordinate clause is the cause
    if surface in _CAUSAL_INV:
        p, q = sub_text, main_text
    else:
        p, q = sub_text, main_text

    # For conditional/causal mark connectives:
    # "Because X, Y" → p=X (cause), q=Y (effect)
    # "If X, Y"      → p=X (condition), q=Y (consequence)
    if len(p.split()) < 2 or len(q.split()) < 2:
        return None

    confidence = 0.6 if polarity == 0 else 1.0
    sentence = doc.text.strip()

    return NLPStatement(
        sentence=sentence,
        p=p, connective=surface, conn_type=conn_type,
        polarity=polarity, q=q, confidence=confidence,
        method="spacy_dep", dep_rel="mark→advcl",
    )


def _extract_via_advmod(doc, conn_start: int, conn_end: int,
                        surface: str, conn_type: str, polarity: int) -> Optional[NLPStatement]:
    """
    Extract arguments when the connective is an `advmod` dependent.

    Pattern: "X rose, however Y remained stagnant."
             advmod→ROOT  (or advmod→main_verb with preceding clause as ccomp)
    """
    conn_token = doc[conn_start]

    if conn_token.dep_ != "advmod":
        return None

    main_verb = conn_token.head

    # Look for a preceding clause: the ccomp, conj, or parataxis child
    # that appears BEFORE the connective
    preceding_clause = None
    for child in main_verb.children:
        if child.dep_ in ("ccomp", "conj", "parataxis") and child.i < conn_start:
            preceding_clause = child
            break

    if preceding_clause is None:
        # Try: main_verb's head might own the preceding clause
        if main_verb.head != main_verb:
            for child in main_verb.head.children:
                if child.i < conn_start and child != main_verb:
                    preceding_clause = child
                    break

    if preceding_clause is None:
        return None

    pre_start, pre_end = _subtree_span(preceding_clause)
    p_text = _span_text(doc, pre_start, pre_end)

    # main clause = main_verb's subtree, minus the connective and preceding clause
    main_start, main_end = _subtree_span(main_verb)
    q_tokens = [t for t in doc[main_start:main_end]
                if (t.i < conn_start or t.i >= conn_end)
                and not (pre_start <= t.i < pre_end)
                and not t.is_punct and not t.is_space]

    if not q_tokens:
        return None

    q_text = " ".join(t.text for t in q_tokens).strip()

    if len(p_text.split()) < 2 or len(q_text.split()) < 2:
        return None

    confidence = 0.6 if polarity == 0 else 1.0
    sentence = doc.text.strip()

    return NLPStatement(
        sentence=sentence,
        p=p_text, connective=surface, conn_type=conn_type,
        polarity=polarity, q=q_text, confidence=confidence,
        method="spacy_dep", dep_rel="advmod",
    )


def _extract_via_root_verb(doc, conn_start: int, conn_end: int,
                           surface: str, conn_type: str, polarity: int) -> Optional[NLPStatement]:
    """
    Extract arguments when the connective IS the ROOT verb (or part of it).

    Pattern: "The fuel hike caused transport costs to rise."
             nsubj←ROOT→ccomp/dobj/xcomp

    The subject subtree is p, the complement subtree is q.
    """
    conn_token = doc[conn_start]

    # The connective should be the root or very close to it
    root = conn_token
    if root.dep_ != "ROOT":
        # Walk up at most 1 level
        if root.head.dep_ == "ROOT":
            root = root.head
        else:
            return None

    # For multi-word connectives like "led to", the root is "led"
    # and "to" is prep — we want the verb

    # Find subject subtree
    subj = None
    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj = child
            break

    if subj is None:
        return None

    # Find complement subtree
    comp = None
    for child in root.children:
        if child.dep_ in ("ccomp", "xcomp", "dobj", "attr"):
            comp = child
            break

    # Also check prep children for "led to X" patterns
    if comp is None:
        for child in root.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        comp = child  # use the prep subtree
                        break
                if comp:
                    break

    if comp is None:
        return None

    subj_start, subj_end = _subtree_span(subj)
    comp_start, comp_end = _subtree_span(comp)

    p_text = _span_text(doc, subj_start, subj_end)
    q_text = _span_text(doc, comp_start, comp_end)

    # For passive + causal-inverse: "protests were caused due to prices"
    if surface in _CAUSAL_INV:
        p_text, q_text = q_text, p_text

    if len(p_text.split()) < 2 or len(q_text.split()) < 2:
        return None

    confidence = 0.6 if polarity == 0 else 1.0
    sentence = doc.text.strip()

    return NLPStatement(
        sentence=sentence,
        p=p_text, connective=surface, conn_type=conn_type,
        polarity=polarity, q=q_text, confidence=confidence,
        method="spacy_dep", dep_rel=f"root_verb({root.dep_})",
    )


def _extract_via_prep(doc, conn_start: int, conn_end: int,
                      surface: str, conn_type: str, polarity: int) -> Optional[NLPStatement]:
    """
    Extract arguments for prepositional connectives like "due to", "owing to".

    Pattern: "The protests were caused due to rising fuel prices."
             prep→ROOT  with pobj subtree as one argument
    """
    conn_token = doc[conn_start]

    if conn_token.dep_ != "prep":
        return None

    head_verb = conn_token.head

    # Find the pobj under the prep chain
    pobj = None
    # "due to X" → due(prep) → to(prep) → X(pobj)
    frontier = [conn_token]
    while frontier:
        node = frontier.pop(0)
        for child in node.children:
            if child.dep_ == "pobj":
                pobj = child
                break
            if child.dep_ == "prep":
                frontier.append(child)
        if pobj:
            break

    if pobj is None:
        return None

    # pobj subtree = the cause
    pobj_start, pobj_end = _subtree_span(pobj)
    cause_text = _span_text(doc, pobj_start, pobj_end)

    # head verb's subject = the effect
    subj = None
    for child in head_verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj = child
            break

    if subj is None:
        return None

    subj_start, subj_end = _subtree_span(subj)
    effect_text = _span_text(doc, subj_start, subj_end)

    # "due to" is causal-inverse: cause is on the right
    p_text, q_text = cause_text, effect_text

    if len(p_text.split()) < 2 or len(q_text.split()) < 2:
        return None

    confidence = 0.6 if polarity == 0 else 1.0
    sentence = doc.text.strip()

    return NLPStatement(
        sentence=sentence,
        p=p_text, connective=surface, conn_type=conn_type,
        polarity=polarity, q=q_text, confidence=confidence,
        method="spacy_dep", dep_rel=f"prep→{head_verb.dep_}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TOP-LEVEL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

_EXTRACTORS = [
    _extract_via_mark,
    _extract_via_advmod,
    _extract_via_root_verb,
    _extract_via_prep,
]


def extract_statements_nlp(text: str) -> list[NLPStatement]:
    """
    Parse text with spaCy and extract (p, connective, q) statements
    using dependency-tree-based argument splitting.

    Returns an empty list if spaCy is unavailable.
    Each sentence is processed independently.
    """
    nlp = _load_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    results = []

    for sent in doc.sents:
        sent_doc = sent.as_doc()
        conn_spans = _find_connective_spans(sent_doc)

        for (cs, ce, surface, conn_type, polarity) in conn_spans:
            # Try each extraction strategy in priority order
            for extractor in _EXTRACTORS:
                stmt = extractor(sent_doc, cs, ce, surface, conn_type, polarity)
                if stmt is not None:
                    results.append(stmt)
                    break   # first successful extractor wins for this connective

    return results
