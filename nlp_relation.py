"""
nlp_relation.py
================
Phase 2 (NLP-model tier) — similarity scoring for adjacent sentence pairs.

The README describes this tier as eventually being a fine-tuned
`xlm-roberta-base` sentence-pair classifier. That needs labelled training
data and a GPU this environment doesn't have, so `ImplicitRelationModel` is
the pluggable interface it will sit behind; `TfidfRelationModel` is a real,
working stand-in — cosine similarity over TF-IDF vectors, using
`tokenizer.word_tokenize` so scoring stays consistent with the rest of the
pipeline (same tokenization tokenizer.py already uses for connective
matching and word counts).

Usage:
    from nlp_relation import TfidfRelationModel
    model = TfidfRelationModel()
    score = model.score_pair(sent1, sent2)
"""
from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tokenizer import word_tokenize


class ImplicitRelationModel(ABC):
    @abstractmethod
    def score_pair(self, sent1: str, sent2: str) -> float:
        """Return a similarity/relatedness score in [0, 1] for the pair."""
        raise NotImplementedError


class TfidfRelationModel(ImplicitRelationModel):
    def score_pair(self, sent1: str, sent2: str) -> float:
        vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
        matrix = vectorizer.fit_transform([sent1, sent2])
        sim = cosine_similarity(matrix[0], matrix[1])[0][0]
        return float(sim)
