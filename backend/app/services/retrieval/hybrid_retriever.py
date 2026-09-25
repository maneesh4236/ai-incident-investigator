"""
Combines vector search (Qdrant) and graph search (Neo4j/NetworkX) into a
single retrieval result that downstream reasoning (RCA, timeline, chat) can
consume without caring which retrieval strategy found what.
"""
from __future__ import annotations

from app.core.logging_config import get_logger
from app.models.schemas import HybridRetrievalResult
from app.services.graph.graph_retriever import GraphRetriever
from app.services.vector.vector_retriever import VectorRetriever

logger = get_logger("retrieval.hybrid_retriever")


class HybridRetriever:
    def __init__(self, vector_retriever: VectorRetriever, graph_retriever: GraphRetriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

    def retrieve(self, investigation_id: str, query: str) -> HybridRetrievalResult:
        logger.info(f"Hybrid retrieval starting for investigation={investigation_id} query='{query}'")

        chunks = self.vector_retriever.retrieve(investigation_id, query)
        graph_context = self.graph_retriever.retrieve(investigation_id, query)

        return HybridRetrievalResult(query=query, chunks=chunks, graph_context=graph_context)
