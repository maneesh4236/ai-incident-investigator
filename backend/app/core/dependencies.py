"""
Wires up service singletons and exposes them as FastAPI dependencies.

Kept in one place so the API layer stays thin (routes just declare which
services they need) and so tests can override individual dependencies
(e.g. swap the real GeminiClient for a stub) without touching route code.
"""
from __future__ import annotations

from functools import lru_cache

from app.repositories.incident_repository import incident_repository
from app.services.agents.investigation_agent import InvestigationAgent
from app.services.graph.graph_builder import GraphBuilder
from app.services.graph.graph_retriever import GraphRetriever
from app.services.graph.neo4j_service import Neo4jService
from app.services.ingestion.chunker import Chunker
from app.services.ingestion.document_loader import DocumentLoader
from app.services.ingestion.entity_extractor import EntityExtractor
from app.services.ingestion.log_loader import LogLoader
from app.services.ingestion.summarizer import Summarizer
from app.services.llm.gemini_client import GeminiClient
from app.services.reasoning.evidence_builder import EvidenceBuilder
from app.services.reasoning.report_generator import ReportGenerator
from app.services.reasoning.root_cause_analyzer import RootCauseAnalyzer
from app.services.reasoning.timeline_builder import TimelineBuilder
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.vector.embedder import Embedder
from app.services.vector.qdrant_service import QdrantService
from app.services.vector.vector_retriever import VectorRetriever


@lru_cache
def get_gemini_client() -> GeminiClient:
    return GeminiClient()


@lru_cache
def get_neo4j_service() -> Neo4jService:
    return Neo4jService()


@lru_cache
def get_qdrant_service() -> QdrantService:
    return QdrantService()


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache
def get_document_loader() -> DocumentLoader:
    return DocumentLoader()


@lru_cache
def get_log_loader() -> LogLoader:
    return LogLoader()


@lru_cache
def get_chunker() -> Chunker:
    return Chunker()


@lru_cache
def get_entity_extractor() -> EntityExtractor:
    return EntityExtractor(llm_client=get_gemini_client())


@lru_cache
def get_summarizer() -> Summarizer:
    return Summarizer(llm_client=get_gemini_client())


@lru_cache
def get_graph_builder() -> GraphBuilder:
    return GraphBuilder(neo4j_service=get_neo4j_service())


@lru_cache
def get_graph_retriever() -> GraphRetriever:
    return GraphRetriever(graph_builder=get_graph_builder())


@lru_cache
def get_vector_retriever() -> VectorRetriever:
    return VectorRetriever(embedder=get_embedder(), qdrant_service=get_qdrant_service())


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        vector_retriever=get_vector_retriever(), graph_retriever=get_graph_retriever()
    )


@lru_cache
def get_evidence_builder() -> EvidenceBuilder:
    return EvidenceBuilder()


@lru_cache
def get_root_cause_analyzer() -> RootCauseAnalyzer:
    return RootCauseAnalyzer(
        graph_builder=get_graph_builder(),
        llm_client=get_gemini_client(),
        evidence_builder=get_evidence_builder(),
    )


@lru_cache
def get_timeline_builder() -> TimelineBuilder:
    return TimelineBuilder(llm_client=get_gemini_client())


@lru_cache
def get_report_generator() -> ReportGenerator:
    return ReportGenerator(llm_client=get_gemini_client())


@lru_cache
def get_investigation_agent() -> InvestigationAgent:
    return InvestigationAgent(
        hybrid_retriever=get_hybrid_retriever(),
        llm_client=get_gemini_client(),
        evidence_builder=get_evidence_builder(),
    )


def get_incident_repository():
    return incident_repository
