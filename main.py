from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator
from store import TripleStore, normalise
from inference import bfs_query
from ingestion import fetch_feed, RSS_FEEDS

app   = FastAPI(title="Tathya inference engine", version="0.1.0")
store = TripleStore()


# ------------------------------------------------------------------ schemas

class Triple(BaseModel):
    p:        str
    polarity: int
    q:        str
    weight:   float = 1.0

    @field_validator("polarity")
    @classmethod
    def polarity_must_be_binary(cls, v):
        if v not in (1, -1):
            raise ValueError("polarity must be 1 or -1")
        return v

    @field_validator("p", "q")
    @classmethod
    def must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("node label must not be empty")
        return v


class QueryRequest(BaseModel):
    p:                 str
    q:                 str
    expected_polarity: int | None = None

    @field_validator("expected_polarity")
    @classmethod
    def validate_expected(cls, v):
        if v is not None and v not in (1, -1):
            raise ValueError("expected_polarity must be 1, -1, or null")
        return v


# ------------------------------------------------------------------ core routes

@app.get("/")
def root():
    return {
        "service": "Tathya inference engine",
        "endpoints": {
            "POST /assert":  "add a (p, polarity, q) triple",
            "POST /query":   "BFS path query from p to q",
            "GET  /triples": "dump all asserted triples",
            "GET  /feed":    "fetch articles from an RSS source",
            "GET  /sources": "list available RSS sources",
            "POST /extract": "extract connectives + statements via NLP",
        }
    }


@app.post("/assert")
def assert_triple(body: Triple):
    return store.assert_triple(
        p=body.p, polarity=body.polarity, q=body.q, weight=body.weight,
    )


@app.post("/query")
def query(body: QueryRequest):
    if not body.p.strip() or not body.q.strip():
        raise HTTPException(status_code=422, detail="p and q must not be empty")

    result = bfs_query(store, body.p, body.q)

    if not result["known"]:
        verdict = "unknown"
    elif body.expected_polarity is None:
        verdict = None
    else:
        bias = result["bias_score"]
        if bias is None:
            verdict = "unknown"
        elif bias > 0 and body.expected_polarity == 1:
            verdict = "agrees"
        elif bias < 0 and body.expected_polarity == -1:
            verdict = "agrees"
        else:
            verdict = "contradicts"

    return {"p": body.p, "q": body.q, **result, "verdict": verdict}


@app.get("/triples")
def list_triples():
    triples = store.all_triples()
    return {"count": len(triples), "triples": triples}


# ------------------------------------------------------------------ ingestion

@app.get("/sources")
def list_sources():
    """List all configured RSS sources."""
    return {"sources": list(RSS_FEEDS.keys())}


@app.get("/feed")
def get_feed(
    source: str = Query(default="onlinekhabar", description="RSS source key"),
    limit:  int = Query(default=20, ge=1, le=50, description="Max articles"),
):
    """
    Fetch articles from an RSS feed — returns headline + body for each.

    GET /feed?source=onlinekhabar&limit=10
    """
    try:
        articles = fetch_feed(source=source, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "source":   source,
        "count":    len(articles),
        "articles": [
            {
                "headline":  a.headline,
                "url":       a.url,
                "published": a.published,
                "body":      a.body,
                "image":     a.image,
            }
            for a in articles
        ],
    }


# ------------------------------------------------------------------ tokenize

class TokenizeRequest(BaseModel):
    headline: str
    body: str

@app.post("/tokenize")
def tokenize(body: TokenizeRequest):
    """
    Tokenize an article and extract (p, connective, q) statements.

    Example:
        {
          "headline": "...",
          "body": "..."
        }
    """
    from tokenizer import tokenize_article
    result = tokenize_article(body.headline, body.body)
    return {
        "headline":    result.headline,
        "token_count": result.token_count,
        "sentences":   result.sentences,
        "statements": [
            {
                "p":          s.p,
                "connective": s.connective,
                "conn_type":  s.conn_type,
                "polarity":   s.polarity,
                "q":          s.q,
                "confidence": s.confidence,
                "method":     s.method,
                "sentence":   s.sentence,
            }
            for s in result.statements
        ],
    }


# ------------------------------------------------------------------ extract

class AnalyseRequest(BaseModel):
    headline: str = ""
    body: str
    source: str = "unknown"

@app.post("/extract")
def extract(req: AnalyseRequest):
    """
    Extract connectives and statements from article text using NLP.

    Returns structured (p, connective, q) triples with dependency-parse
    metadata.  Uses spaCy dep-parse for English text and falls back to
    the regex pipeline for Nepali text.

    Example body:
        {
          "headline": "Fuel prices rise",
          "body":     "The fuel price hike caused transport costs to rise.",
          "source":   "onlinekhabar"
        }
    """
    from tokenizer import tokenize_article

    if not req.body.strip():
        raise HTTPException(status_code=422, detail="body must not be empty")

    tokenized = tokenize_article(req.headline, req.body)

    return {
        "headline":        req.headline,
        "source":          req.source,
        "token_count":     tokenized.token_count,
        "sentence_count":  len(tokenized.sentences),
        "statement_count": len(tokenized.statements),
        "statements": [
            {
                "p":          s.p,
                "connective": s.connective,
                "conn_type":  s.conn_type,
                "polarity":   s.polarity,
                "q":          s.q,
                "confidence": s.confidence,
                "method":     s.method,
                "dep_rel":    getattr(s, 'dep_rel', ''),
                "sentence":   s.sentence,
            }
            for s in tokenized.statements
        ],
    }


# ------------------------------------------------------------------ analyse

@app.post("/analyse")
def analyse(req: AnalyseRequest):
    """
    Full pipeline for one article:

      1. Tokenize headline + body → sentences + Statement list
      2. For each Statement:
           a. Normalise p and q to node keys
           b. BFS query the triple store for any path p ⇒ q
           c. If a path exists  → attach bias_score + chain (KNOWN)
           d. If no path exists → store the triple as a new seed fact (NEW)
      3. Return per-statement verdicts + an aggregate article bias score

    The aggregate bias is the mean of all known-statement scores.
    New facts do not contribute to the bias (no prior evidence).

    Example body:
        {
          "headline": "Fuel prices rise",
          "body":     "The fuel price hike caused transport costs to rise.",
          "source":   "onlinekhabar"
        }
    """
    from tokenizer import tokenize_article

    if not req.body.strip():
        raise HTTPException(status_code=422, detail="body must not be empty")

    # ── 1. extract statements ────────────────────────────────────────────────
    tokenized = tokenize_article(req.headline, req.body)

    results = []
    known_scores = []

    for stmt in tokenized.statements:

        # ── 2. normalise to node keys (basic: lowercase + underscores) ───────
        p_key = normalise(stmt.p)
        q_key = normalise(stmt.q)

        # ── 3. query the triple store ────────────────────────────────────────
        kb = bfs_query(store, p_key, q_key)

        if kb["known"]:
            # path found — score it
            bias = kb["bias_score"]
            known_scores.append(bias)

            # does the article's polarity agree with the KB?
            if stmt.polarity == 0:
                verdict = "hedged"
            elif bias > 0 and stmt.polarity == 1:
                verdict = "supported"
            elif bias < 0 and stmt.polarity == -1:
                verdict = "supported"
            else:
                verdict = "contradicted"

            results.append({
                "status":     "known",
                "verdict":    verdict,
                "p":          stmt.p,
                "connective": stmt.connective,
                "conn_type":  stmt.conn_type,
                "polarity":   stmt.polarity,
                "q":          stmt.q,
                "bias_score": round(bias, 4),
                "method":     stmt.method,
                "paths":      kb["paths"],
                "sentence":   stmt.sentence,
            })

        else:
            # ── 4. no path — ingest as new seed fact ─────────────────────────
            # hedged statements (polarity=0) are not ingested as hard edges
            if stmt.polarity != 0:
                store.assert_triple(
                    p=p_key,
                    polarity=stmt.polarity,
                    q=q_key,
                    weight=0.5,           # seed weight — lower than corroborated
                )

            results.append({
                "status":     "new",
                "verdict":    "no_prior_data",
                "p":          stmt.p,
                "connective": stmt.connective,
                "conn_type":  stmt.conn_type,
                "polarity":   stmt.polarity,
                "q":          stmt.q,
                "bias_score": None,
                "method":     stmt.method,
                "paths":      [],
                "sentence":   stmt.sentence,
            })

    # ── 5. aggregate bias across known statements ────────────────────────────
    article_bias = (
        round(sum(known_scores) / len(known_scores), 4)
        if known_scores else None
    )

    return {
        "headline":        req.headline,
        "source":          req.source,
        "token_count":     tokenized.token_count,
        "sentence_count":  len(tokenized.sentences),
        "statement_count": len(results),
        "known_count":     len(known_scores),
        "new_count":       len(results) - len(known_scores),
        "article_bias":    article_bias,
        "statements":      results,
    }
