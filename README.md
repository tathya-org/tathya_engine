# Tathya — Logical Truth Engine for Nepali News

Tathya (तथ्य — *fact*) is a news aggregator that analyses articles for logical consistency. When a user opens an article, statements are extracted as implication triples (`p ⇒ q`) and cross-referenced against a knowledge graph of prior assertions. Each statement is scored for how well it agrees or conflicts with the accumulated body of evidence, and highlighted inline in green, red, or grey.

---

## The problem

High-frequency news environments in language markets like Nepali media make it practically impossible for a reader to track whether a claim made today contradicts something reported last week by a different outlet.

---

## How it works

```
article (headline + body)
        │
        ▼
┌─────────────────────────┐
│   Statement extractor   │  NLP pipeline: connective detection →
│   (backend)             │  argument splitting → polarity tagging
└────────────┬────────────┘
             │  list of (p, connective, q, polarity, confidence)
             ▼
┌─────────────────────────┐
│   Entity normaliser     │  canonical node keys:
│                         │  "fuel hike" == "petrol price rise"
└────────────┬────────────┘
             │
      ┌──────┴──────┐
      │             │
  known?         unknown?
      │             │
      ▼             ▼
┌──────────┐  ┌──────────────┐
│  Scorer  │  │  New fact    │
│          │  │  ingestion   │
│ weighted │  │  seeded by   │
│ polarity │  │  source cred │
└────┬─────┘  └──────────────┘
     │
     ▼
truth score + evidence chain
returned to frontend per statement
```

### Statement extraction

The extractor uses a two-phase pipeline:

**Phase 1 — Explicit connectives.** A lexicon of connective words (`"caused"`, `"led to"`, `"को कारण"`, `"त्यसैले"`, etc.) is matched against spaCy's token list. The connective's position in the dependency tree is used as a pivot to extract the antecedent (`p`) and consequent (`q`) clause spans via subtree walking rather than naive string splitting. Each connective type maps to a polarity: causal and conditional connectives give `+1`; adversative connectives give `-1`; hedged language (`"may"`, `"allegedly"`, `"सक्छ"`) gives `?`.

**Phase 2 — Implicit relations (planned).** Adjacent sentence pairs with no surface connective are passed to a fine-tuned `xlm-roberta-base` sentence-pair classifier to detect implicit causal and conditional relations.

### Knowledge graph

Extracted triples are stored as directed edges in a triple store:

```
(p_key) --[conn_type, polarity, confidence, sources]--> (q_key)
```

Querying is a bounded BFS (depth ≤ 5) over this graph to find any transitive path between `p` and `q`. The cumulative polarity of a chain is the product of its edge polarities — two negatives correctly produce a positive.

### Source credibility

Each source outlet maintains a credibility score initialised at `0.5` and updated whenever a contradiction resolves — the minority-side source receives a small penalty, the majority side a small boost. This is self-correcting: no manual labelling is required.

---

## Repository layout (planned)

```
tathya_engine/
   ├── parser.py            # connective definitions, types, polarity
   ├── extractor.py          # spaCy pipeline + argument splitter
   ├── normaliser.py         # entity canonicalisation
   ├── inference.py          # BFS transitive path query + scoring
   └── tests/
       ├── test_extractor.py
       └── fixtures/         # sample Nepali + English sentences
```

---
