"""
Builds a normalized list of `Evidence` objects from hybrid retrieval
results, used to back RCA findings and chat answers with traceable sources.
"""
from __future__ import annotations

from typing import List

from app.models.schemas import Evidence, HybridRetrievalResult


class EvidenceBuilder:
    def build(self, retrieval: HybridRetrievalResult, document_names: dict[str, str] | None = None) -> List[Evidence]:
        document_names = document_names or {}
        evidence: List[Evidence] = []

        for retrieved in retrieval.chunks:
            chunk = retrieved.chunk
            evidence.append(
                Evidence(
                    text=chunk.text[:400],
                    source_document=document_names.get(chunk.document_id, chunk.document_id),
                    chunk_id=chunk.id,
                    relevance=retrieved.score,
                )
            )

        # Sort strongest evidence first so report/chat surfaces the best support.
        evidence.sort(key=lambda e: e.relevance, reverse=True)
        return evidence
