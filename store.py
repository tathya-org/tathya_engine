from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, XSD
from typing import Optional

TATHYA = Namespace("http://tathya.local/")

# polarity URIs
ENABLES  = TATHYA.enables   # +1  p => q
INHIBITS = TATHYA.inhibits  # -1  p => ~q

POLARITY_URI = {
    1:  ENABLES,
    -1: INHIBITS,
}
URI_POLARITY = {
    ENABLES:  1,
    INHIBITS: -1,
}


def _node(term: str) -> URIRef:
    """Slugify a normalised noun into a URI."""
    slug = term.strip().lower().replace(" ", "_")
    return TATHYA[slug]


class TripleStore:
    def __init__(self):
        self.g = Graph()
        self.g.bind("tathya", TATHYA)

    # ------------------------------------------------------------------ write

    def assert_triple(self, p: str, polarity: int, q: str, weight: float = 1.0) -> dict:
        """
        Assert (p) -[polarity]-> (q) into the store.
        Duplicate assertions increase the weight on the existing edge.
        """
        if polarity not in POLARITY_URI:
            raise ValueError(f"polarity must be +1 or -1, got {polarity}")

        p_uri  = _node(p)
        q_uri  = _node(q)
        rel    = POLARITY_URI[polarity]

        # weight predicate: tathya:weight
        w_pred = TATHYA.weight

        # check if edge already exists
        existing = any(True for _ in self.g.triples((p_uri, rel, q_uri)))
        if existing:
            # edge exists as a reified statement — update weight
            for stmt in self.g.subjects(RDF.type, RDF.Statement):
                s = self.g.value(stmt, RDF.subject)
                p_ = self.g.value(stmt, RDF.predicate)
                o = self.g.value(stmt, RDF.object)
                if s == p_uri and p_ == rel and o == q_uri:
                    old_w = float(self.g.value(stmt, w_pred) or 1.0)
                    self.g.set((stmt, w_pred, Literal(old_w + weight, datatype=XSD.float)))
                    return {"status": "updated", "p": p, "q": q, "polarity": polarity, "weight": old_w + weight}

        # new edge — reify it so we can attach weight
        self.g.add((p_uri, rel, q_uri))
        stmt_node = TATHYA[f"stmt_{p_uri.split('/')[-1]}_{rel.split('/')[-1]}_{q_uri.split('/')[-1]}"]
        self.g.add((stmt_node, RDF.type,      RDF.Statement))
        self.g.add((stmt_node, RDF.subject,   p_uri))
        self.g.add((stmt_node, RDF.predicate, rel))
        self.g.add((stmt_node, RDF.object,    q_uri))
        self.g.add((stmt_node, w_pred,        Literal(weight, datatype=XSD.float)))

        return {"status": "created", "p": p, "q": q, "polarity": polarity, "weight": weight}

    # ------------------------------------------------------------------ read

    def get_edges(self, p: str) -> list[dict]:
        """Return all direct edges from node p."""
        p_uri = _node(p)
        edges = []
        for rel in (ENABLES, INHIBITS):
            for q_uri in self.g.objects(p_uri, rel):
                weight = self._get_weight(p_uri, rel, q_uri)
                edges.append({
                    "q":       str(q_uri).split("/")[-1],
                    "polarity": URI_POLARITY[rel],
                    "weight":  weight,
                })
        return edges

    def _get_weight(self, p_uri: URIRef, rel: URIRef, q_uri: URIRef) -> float:
        for stmt in self.g.subjects(RDF.type, RDF.Statement):
            if (
                self.g.value(stmt, RDF.subject)   == p_uri and
                self.g.value(stmt, RDF.predicate) == rel   and
                self.g.value(stmt, RDF.object)    == q_uri
            ):
                return float(self.g.value(stmt, TATHYA.weight) or 1.0)
        return 1.0

    def all_triples(self) -> list[dict]:
        """Dump every asserted triple (for debugging)."""
        out = []
        for rel in (ENABLES, INHIBITS):
            for p_uri, _, q_uri in self.g.triples((None, rel, None)):
                weight = self._get_weight(p_uri, rel, q_uri)
                out.append({
                    "p":       str(p_uri).split("/")[-1],
                    "polarity": URI_POLARITY[rel],
                    "q":       str(q_uri).split("/")[-1],
                    "weight":  weight,
                })
        return out
