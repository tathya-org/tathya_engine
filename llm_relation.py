"""
llm_relation.py
================
Phase 2 (LLM tier) — implicit relation extraction via an LLM, for the
adjacent sentence pairs the rule-based (implicit_rules.py) and TF-IDF
(nlp_relation.py) tiers are least confident about.

`LLMClient` is the pluggable interface. `AnthropicLLMClient` is a real HTTP
implementation (needs ANTHROPIC_API_KEY + network — neither available in
this sandbox, so it's untested here but is genuine, runnable code).
`MockLLMClient` returns deterministic JSON so `LLMRelationExtractor` and
the benchmark harness are fully testable without a live API.

Output uses the same (conn_type, polarity) shape as
`implicit_rules.ImplicitResult` / `tokenizer.Statement`, so all three tiers
are interchangeable from the caller's point of view.
"""
import hashlib
import json
import os
from abc import ABC, abstractmethod

import requests

from implicit_rules import ImplicitResult

PROMPT_TEMPLATE = """Classify the discourse relation between these two sentences.
Respond with ONLY a JSON object: {{"conn_type": "CAUSAL"|"CONDITIONAL"|"ADVERSATIVE"|"HEDGED"|"NONE", "polarity": 1|-1|0, "confidence": 0.0-1.0}}

Sentence 1: {sent1}
Sentence 2: {sent2}"""

_VALID_CONN_TYPES = {"CAUSAL", "CONDITIONAL", "ADVERSATIVE", "HEDGED", "NONE"}


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's raw text response to `prompt`."""
        raise NotImplementedError


class AnthropicLLMClient(LLMClient):
    """Real HTTP client against api.anthropic.com. Needs network + API key."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

    def complete(self, prompt: str) -> str:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(
            block["text"] for block in data["content"] if block.get("type") == "text"
        )


class MockLLMClient(LLMClient):
    """Deterministic mock: same prompt always returns the same response.

    Response is picked from `canned_responses` by exact prompt match if
    given, otherwise deterministically derived from a hash of the prompt so
    repeated benchmark runs are reproducible.
    """

    _DEFAULT_TYPES = ["CAUSAL", "ADVERSATIVE", "HEDGED", "NONE"]
    _POLARITY_BY_TYPE = {"CAUSAL": 1, "ADVERSATIVE": -1, "HEDGED": 0, "NONE": 0}

    def __init__(self, canned_responses: dict[str, str] | None = None):
        self.canned_responses = canned_responses or {}

    def complete(self, prompt: str) -> str:
        if prompt in self.canned_responses:
            return self.canned_responses[prompt]
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        conn_type = self._DEFAULT_TYPES[int(digest, 16) % len(self._DEFAULT_TYPES)]
        confidence = (int(digest[:8], 16) % 100) / 100
        return json.dumps({
            "conn_type": conn_type,
            "polarity": self._POLARITY_BY_TYPE[conn_type],
            "confidence": confidence,
        })


class LLMRelationExtractor:
    def __init__(self, client: LLMClient):
        self.client = client

    def extract(self, sent1: str, sent2: str) -> ImplicitResult:
        prompt = PROMPT_TEMPLATE.format(sent1=sent1, sent2=sent2)
        raw = self.client.complete(prompt)
        try:
            parsed = json.loads(raw)
            conn_type = parsed.get("conn_type", "NONE")
            if conn_type not in _VALID_CONN_TYPES:
                conn_type = "NONE"
            return ImplicitResult(
                conn_type=conn_type,
                polarity=int(parsed.get("polarity", 0)),
                confidence=float(parsed.get("confidence", 0.0)),
                p=sent1,
                q=sent2,
                evidence=[raw],
                method="llm",
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return ImplicitResult(
                conn_type="NONE", polarity=0, confidence=0.0,
                p=sent1, q=sent2, evidence=[raw], method="llm",
            )
