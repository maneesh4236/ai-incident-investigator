"""
Test isolation:
  * Environment variables are set BEFORE the app is imported, so a local
    backend/.env (which may contain real credentials) can never be used:
    process env vars take precedence over the .env file in pydantic-settings.
  * `google.genai.Client` is replaced by `FakeGenaiClient`; no test can reach
    the Gemini API. Tests script responses / failures per call.
  * Neo4j is disabled, Qdrant runs in memory, and the embedding model is a
    deterministic hashing stub (no model download).
"""
from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="aetherlog-tests-")
os.environ.update(
    {
        "GEMINI_API_KEY": "test-key-not-real",
        "GEMINI_MODEL": "fake-model",
        "EMBEDDING_MODEL": "fake-embedder",
        "EMBEDDING_DIM": "64",
        "QDRANT_MODE": "memory",
        "NEO4J_URI": "bolt://127.0.0.1:1",
        "UPLOAD_DIR": os.path.join(_TMP, "uploads"),
        "EMBEDDER_WARMUP_ON_STARTUP": "false",
        "GEMINI_BACKOFF_BASE_SECONDS": "0.01",
        "GEMINI_BACKOFF_MAX_SECONDS": "0.02",
        "GEMINI_TIMEOUT_SECONDS": "5",
    }
)

import hashlib  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
from typing import Any, Callable, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import google.genai as _genai  # noqa: E402


class FakeResponse:
    def __init__(self, text: str, prompt_tokens: int = 0, output_tokens: int = 0):
        self.text = text
        self.prompt_feedback = None
        self.usage_metadata = type(
            "Usage", (), {"prompt_token_count": prompt_tokens, "candidates_token_count": output_tokens}
        )()


class FakeModels:
    def __init__(self, owner: "FakeGenaiClient"):
        self.owner = owner

    def generate_content(self, model: str, contents: str, config: Any = None):
        return self.owner._respond(contents, config)


class FakeGenaiClient:
    """Records every API attempt. `script` is a list of callables/values consumed
    per attempt; when exhausted, `default` is used."""

    instances: List["FakeGenaiClient"] = []

    def __init__(self, *args, **kwargs):
        self.models = FakeModels(self)
        self.calls: List[dict] = []
        self.script: List[Any] = []
        self.default: Callable[[str, Any], Any] = default_responder
        self._lock = threading.Lock()
        FakeGenaiClient.instances.append(self)

    def _respond(self, contents: str, config: Any):
        system = getattr(config, "system_instruction", None) or ""
        with self._lock:
            self.calls.append({"prompt": contents, "system": system})
            action = self.script.pop(0) if self.script else self.default
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            action = action(contents, config)
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, FakeResponse):
            return action
        text = action if isinstance(action, str) else json.dumps(action)
        return FakeResponse(text, prompt_tokens=len(contents) // 4, output_tokens=len(text) // 4)


def default_responder(contents: str, config: Any):
    system = getattr(config, "system_instruction", None) or ""
    if "Incident Investigator answering" in system:
        return {"answer": "stub answer", "claims": [], "referenced_entities": [], "evidence_ids": []}
    return {
        "root_cause": {"statement": "stub root cause", "type": "UNKNOWN", "evidence_ids": []},
        "cause_chain": [],
        "claims": [],
        "timeline": [],
        "affected_systems": [],
        "executive_summary": "stub summary",
        "recommendations": ["stub recommendation"],
        "insufficient_evidence": [],
        "confidence": 0.2,
    }


_genai.Client = FakeGenaiClient  # type: ignore[assignment]

# Neo4j: fail fast (in-memory graph fallback).
import app.services.graph.neo4j_service as _neo4j_service  # noqa: E402


class _NoNeo4j:
    @staticmethod
    def driver(*args, **kwargs):
        raise ConnectionError("Neo4j disabled in tests")


_neo4j_service.GraphDatabase = _NoNeo4j  # type: ignore[assignment]


class FakeEmbeddingModel:
    """Deterministic bag-of-words hashing embedder (64 dims)."""

    max_seq_length = 256

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        out = np.zeros((len(batch), 64), dtype=np.float32)
        for i, text in enumerate(batch):
            for token in str(text).lower().split():
                out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % 64] += 1.0
            norm = np.linalg.norm(out[i]) or 1.0
            out[i] /= norm
        return out[0] if single else out


from app.services.vector.embedder import Embedder  # noqa: E402

Embedder._model = FakeEmbeddingModel()


@pytest.fixture
def gemini():
    """The fake genai client behind the app's GeminiClient singleton (reset per test)."""
    from app.core.dependencies import get_gemini_client

    client = get_gemini_client()
    fake: FakeGenaiClient = client._client
    fake.calls.clear()
    fake.script.clear()
    fake.default = default_responder
    return fake


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def gemini_error(code: int, message: str = "error", details: Optional[dict] = None) -> BaseException:
    """Builds a google-genai APIError subclass instance without an HTTP response."""
    from google.genai import errors

    cls = errors.ClientError if 400 <= code < 500 else errors.ServerError
    exc = cls.__new__(cls)
    Exception.__init__(exc, f"{code} {message}")
    exc.code = code
    exc.status = message
    exc.message = message
    exc.details = details or {"error": {"code": code, "message": message}}
    exc.response = None
    return exc
