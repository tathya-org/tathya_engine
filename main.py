from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator
from store import TripleStore
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
