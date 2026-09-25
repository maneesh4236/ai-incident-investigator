"""
Vector-side retrieval: embeds a query and searches Qdrant for the most
semantically similar chunks within an investigation.

This is the "A. Vector Search" half of hybrid retrieval.
"""
from __future__ import annotations

from typing import List

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import RetrievedChunk
from app.services.vector.embedder import Embedder
from app.services.vector.qdrant_service import QdrantService

logger = get_logger("vector.vector_retriever")


class VectorRetriever:
    def __init__(self, embedder: Embedder | None = None, qdrant_service: QdrantService | None = None):
        self.embedder = embedder or Embedder()
        self.qdrant = qdrant_service or QdrantService()
        self.settings = get_settings()

    def retrieve(self, investigation_id: str, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        top_k = top_k or self.settings.TOP_K_VECTOR
        query_vector = self.embedder.embed_query(query)
        results = self.qdrant.search(investigation_id, query_vector, top_k)
        logger.info(f"Vector retrieval for '{query}' returned {len(results)} chunks")
        return [RetrievedChunk(chunk=chunk, score=score) for chunk, score in results]
