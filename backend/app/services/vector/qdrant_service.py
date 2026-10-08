"""
Thin wrapper around the Qdrant client: collection lifecycle, batched upserts,
rollback by document, and filtered similarity search scoped to a single
investigation.

QDRANT_MODE selects the backend: "local" (embedded, on disk at
QDRANT_LOCAL_PATH - the previous hard-coded behaviour), "server"
(QDRANT_HOST/QDRANT_PORT, e.g. docker-compose) or "memory" (tests).
Embedded mode is not thread-safe, so client calls are serialized.
"""
from __future__ import annotations

import threading
import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.metrics import current_metrics
from app.models.schemas import Chunk

logger = get_logger("vector.qdrant_service")


class QdrantService:
    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self._client: QdrantClient | None = None
        self._lock = threading.Lock()
        # In-memory fallback store so the demo still works if Qdrant is down.
        self._fallback_store: dict[str, list[tuple[Chunk, List[float]]]] = {}

        mode = settings.QDRANT_MODE.lower()
        try:
            if mode == "server":
                self._client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=10)
                where = f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
            elif mode == "memory":
                self._client = QdrantClient(":memory:")
                where = "in-process memory"
            else:
                self._client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
                where = f"embedded storage at {settings.QDRANT_LOCAL_PATH}"
            self._ensure_collection()
            logger.info(f"Qdrant ready ({mode}: {where})")
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
        metrics = current_metrics()

        if not self._client:
            with self._lock:
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
                    "metadata": chunk.metadata,
                },
            )
            for chunk in chunks
            if chunk.embedding
        ]
        batch = max(1, self.settings.QDRANT_UPSERT_BATCH)
        for start in range(0, len(points), batch):
            with self._lock:
                self._client.upsert(collection_name=self.settings.QDRANT_COLLECTION, points=points[start : start + batch])
            if metrics is not None:
                metrics.incr("qdrant_ops")
        if metrics is not None:
            metrics.incr("qdrant_points", len(points))

    def delete_by_document(self, investigation_id: str, document_id: str) -> None:
        """Rollback for a failed upload."""
        if not self._client:
            with self._lock:
                bucket = self._fallback_store.get(investigation_id, [])
                self._fallback_store[investigation_id] = [(c, v) for c, v in bucket if c.document_id != document_id]
            return
        try:
            with self._lock:
                self._client.delete(
                    collection_name=self.settings.QDRANT_COLLECTION,
                    points_selector=qmodels.FilterSelector(
                        filter=qmodels.Filter(
                            must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
                        )
                    ),
                )
        except Exception as exc:  # rollback is best effort
            logger.warning(f"Qdrant rollback for document {document_id} failed: {exc}")

    def search(self, investigation_id: str, query_vector: List[float], top_k: int) -> List[tuple[Chunk, float]]:
        metrics = current_metrics()
        if metrics is not None:
            metrics.incr("qdrant_ops")
        if not self._client:
            return self._fallback_search(investigation_id, query_vector, top_k)

        with self._lock:
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
                metadata=payload.get("metadata") or {},
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
