"""
Thin wrapper around the Qdrant client: collection lifecycle, upserts, and
filtered similarity search scoped to a single investigation.
"""
from __future__ import annotations

import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import Chunk

logger = get_logger("vector.qdrant_service")


class QdrantService:
    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self._client: QdrantClient | None = None
        # In-memory fallback store so the demo still works if Qdrant is down.
        self._fallback_store: dict[str, list[tuple[Chunk, List[float]]]] = {}

        try:
            #self._client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5)
            self._client = QdrantClient(path="./qdrant_data")
            self._ensure_collection()
            logger.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        except Exception as exc:  # pragma: no cover - depends on infra availability
            logger.warning(f"Qdrant unavailable ({exc}); using in-memory fallback vector store")
            self._client = None

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self.settings.QDRANT_COLLECTION not in collections:
            self._client.create_collection(
                collection_name=self.settings.QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=self.settings.EMBEDDING_DIM, distance=qmodels.Distance.COSINE
                ),
            )

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        if not self._client:
            bucket = self._fallback_store.setdefault(chunks[0].investigation_id, [])
            bucket.extend((c, c.embedding or []) for c in chunks)
            return

        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "investigation_id": chunk.investigation_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk in chunks
            if chunk.embedding
        ]
        if points:
            self._client.upsert(collection_name=self.settings.QDRANT_COLLECTION, points=points)

    def search(self, investigation_id: str, query_vector: List[float], top_k: int) -> List[tuple[Chunk, float]]:
        if not self._client:
            return self._fallback_search(investigation_id, query_vector, top_k)

        results = self._client.search(
            collection_name=self.settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="investigation_id", match=qmodels.MatchValue(value=investigation_id))]
            ),
            limit=top_k,
        )
        output: List[tuple[Chunk, float]] = []
        for point in results:
            payload = point.payload
            chunk = Chunk(
                id=payload["chunk_id"],
                document_id=payload["document_id"],
                investigation_id=payload["investigation_id"],
                text=payload["text"],
                chunk_index=payload["chunk_index"],
            )
            output.append((chunk, point.score))
        return output

    def _fallback_search(self, investigation_id: str, query_vector: List[float], top_k: int):
        import numpy as np

        bucket = self._fallback_store.get(investigation_id, [])
        if not bucket:
            return []
        q = np.array(query_vector)
        scored = []
        for chunk, vector in bucket:
            if not vector:
                continue
            v = np.array(vector)
            denom = (np.linalg.norm(q) * np.linalg.norm(v)) or 1e-9
            score = float(np.dot(q, v) / denom)
            scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
