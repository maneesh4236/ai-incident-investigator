"""
Graph-side retrieval: given a natural-language question, finds the most
relevant entities, their relationships, and short causal paths.

This is the "B. Graph Search" half of hybrid retrieval described in the
AetherLog-style architecture.
"""
from __future__ import annotations

import re
from typing import List

from app.core.logging_config import get_logger
from app.models.schemas import GraphEntity, GraphRelationship, RetrievedGraphContext
from app.services.graph.graph_builder import GraphBuilder

logger = get_logger("graph.graph_retriever")

_STOPWORDS = {
    "why", "did", "the", "a", "an", "is", "are", "was", "were", "fail",
    "failed", "what", "how", "when", "which", "service", "system", "show",
    "related", "incidents", "evidence", "supports", "this",
}


class GraphRetriever:
    def __init__(self, graph_builder: GraphBuilder):
        self.graph_builder = graph_builder

    def retrieve(self, investigation_id: str, query: str, max_hops: int = 2) -> RetrievedGraphContext:
        entities, relationships = self.graph_builder.export(investigation_id)
        keywords = self._extract_keywords(query)

        matched_entities = [
            e for e in entities if any(kw in e.name.lower() for kw in keywords)
        ]
        if not matched_entities:
            # Fall back to the most "central" nodes so the LLM still gets context.
            central_names = set(self.graph_builder.centrality_ranked_nodes(investigation_id)[:5])
            matched_entities = [e for e in entities if e.name in central_names]

        matched_names = {e.name for e in matched_entities}
        related_relationships = [
            r for r in relationships if r.source in matched_names or r.target in matched_names
        ]

        paths: List[List[str]] = []
        for entity in matched_entities:
            causes = self.graph_builder.upstream_causes(investigation_id, entity.name, max_hops=max_hops)
            if causes:
                paths.append([*causes[::-1], entity.name])

        logger.info(
            f"Graph retrieval for '{query}' -> {len(matched_entities)} entities, "
            f"{len(related_relationships)} relationships, {len(paths)} causal paths"
        )
        return RetrievedGraphContext(
            entities=matched_entities, relationships=related_relationships, paths=paths
        )

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        words = re.findall(r"[a-zA-Z0-9\-]+", query.lower())
        return [w for w in words if w not in _STOPWORDS and len(w) > 2]
