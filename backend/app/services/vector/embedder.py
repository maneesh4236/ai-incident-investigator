"""
Wraps the BAAI/bge-large-en-v1.5 sentence-transformer model for embedding
chunks and queries.

The model is loaded lazily (on first use) since it's the heaviest import in
the service and many API paths (e.g. graph-only endpoints) never need it.
"""
from __future__ import annotations

from typing import List

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger("vector.embedder")

# bge models recommend prefixing retrieval queries with an instruction.
_QUERY_INSTRUCTION = "Represent this question for retrieving supporting incident evidence: "


class Embedder:
    _model = None  # class-level cache so we only ever load the model once

    def __init__(self):
        self.settings = get_settings()

    def _load_model(self):
        if Embedder._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model {self.settings.EMBEDDING_MODEL} (first use)...")
            Embedder._model = SentenceTransformer(self.settings.EMBEDDING_MODEL)
        return Embedder._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> List[float]:
        model = self._load_model()
        vector = model.encode(_QUERY_INSTRUCTION + query, normalize_embeddings=True)
        return vector.tolist()
