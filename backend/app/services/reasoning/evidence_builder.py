"""
Builds a normalized list of `Evidence` objects, used to back RCA findings and
chat answers with traceable sources.

* `from_pack` - log investigations: one Evidence per selected *complete*
  event, carrying its event id, group id, timestamp, level and occurrences.
* `build`     - legacy retrieval-chunk path (non-log documents). Chunk text is
  passed through whole (the former 400-character cut is gone); prompt size is
  controlled by the evidence selector's token budget instead.
"""
from __future__ import annotations

from typing import Dict, List

from app.models.schemas import Evidence, HybridRetrievalResult
from app.services.reasoning.evidence_selector import EvidencePack


class EvidenceBuilder:
    def build(self, retrieval: HybridRetrievalResult, document_names: dict[str, str] | None = None) -> List[Evidence]:
        document_names = document_names or {}
        evidence: List[Evidence] = []

        for retrieved in retrieval.chunks:
            chunk = retrieved.chunk
            evidence.append(
                Evidence(
                    text=chunk.text,
                    source_document=document_names.get(chunk.document_id, chunk.document_id),
                    chunk_id=chunk.id,
                    relevance=retrieved.score,
                    group_id=(chunk.metadata or {}).get("group_id"),
                )
            )

        # Sort strongest evidence first so report/chat surfaces the best support.
        evidence.sort(key=lambda e: e.relevance, reverse=True)
        return evidence

    @staticmethod
    def from_pack(
        pack: EvidencePack, group_point_ids: Dict[str, str], document_names: Dict[str, str] | None = None
    ) -> List[Evidence]:
        return pack.to_evidence(group_point_ids, document_names or {})
