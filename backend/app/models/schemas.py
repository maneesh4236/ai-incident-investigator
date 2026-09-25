"""
Shared Pydantic models (DTOs) used across API, services, and repositories.

Keeping these in one module gives every layer a single source of truth for
the shape of an "incident", a "chunk", a "graph entity", etc.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class EntityType(str, Enum):
    SERVICE = "SERVICE"
    ERROR = "ERROR"
    COMPONENT = "COMPONENT"
    DATABASE = "DATABASE"
    API = "API"
    EVENT = "EVENT"
    SYSTEM = "SYSTEM"


class RelationType(str, Enum):
    CAUSES = "CAUSES"
    DEPENDS_ON = "DEPENDS_ON"
    RELATED_TO = "RELATED_TO"
    CONTAINS = "CONTAINS"
    TRIGGERS = "TRIGGERS"


class DocumentType(str, Enum):
    LOG = "LOG"
    PDF = "PDF"
    INCIDENT_REPORT = "INCIDENT_REPORT"
    RUNBOOK = "RUNBOOK"
    ARCHITECTURE_DOC = "ARCHITECTURE_DOC"
    TEXT = "TEXT"


class InvestigationStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class UploadedDocument(BaseModel):
    id: str
    filename: str
    doc_type: DocumentType
    investigation_id: str
    stored_path: str
    size_bytes: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    id: str
    document_id: str
    investigation_id: str
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


# --------------------------------------------------------------------------- #
# Knowledge graph
# --------------------------------------------------------------------------- #
class GraphEntity(BaseModel):
    id: str
    name: str
    type: EntityType
    investigation_id: str
    source_chunk_ids: List[str] = Field(default_factory=list)


class GraphRelationship(BaseModel):
    id: str
    source: str  # entity name
    target: str  # entity name
    type: RelationType
    investigation_id: str
    confidence: float = 0.7
    evidence_chunk_ids: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    entities: List[GraphEntity]
    relationships: List[GraphRelationship]


class KnowledgeGraphResponse(BaseModel):
    investigation_id: str
    nodes: List[GraphEntity]
    edges: List[GraphRelationship]


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class RetrievedGraphContext(BaseModel):
    entities: List[GraphEntity]
    relationships: List[GraphRelationship]
    paths: List[List[str]] = Field(default_factory=list)


class HybridRetrievalResult(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    graph_context: RetrievedGraphContext


# --------------------------------------------------------------------------- #
# Root cause analysis
# --------------------------------------------------------------------------- #
class Evidence(BaseModel):
    text: str
    source_document: str
    chunk_id: str
    relevance: float = 0.0


class RootCauseResult(BaseModel):
    root_cause: str
    cause_chain: List[str]
    confidence_score: float
    evidence: List[Evidence]
    affected_systems: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #
class TimelineEvent(BaseModel):
    timestamp: Optional[str] = None
    order: int
    title: str
    description: str
    severity: str = "info"  # info | warning | critical
    source_chunk_ids: List[str] = Field(default_factory=list)


class Timeline(BaseModel):
    investigation_id: str
    events: List[TimelineEvent]


# --------------------------------------------------------------------------- #
# RCA Report
# --------------------------------------------------------------------------- #
class RCAReport(BaseModel):
    investigation_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    executive_summary: str
    root_cause: RootCauseResult
    timeline: Timeline
    affected_systems: List[str]
    recommendations: List[str]
    confidence: float


# --------------------------------------------------------------------------- #
# Investigation
# --------------------------------------------------------------------------- #
class Investigation(BaseModel):
    id: str
    title: str
    status: InvestigationStatus = InvestigationStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    document_ids: List[str] = Field(default_factory=list)
    report: Optional[RCAReport] = None


class InvestigateRequest(BaseModel):
    investigation_id: str
    question: Optional[str] = Field(
        default=None,
        description="Optional focusing question, e.g. 'Why did payment service fail?'",
    )


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str  # user | assistant
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    investigation_id: str
    message: str
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    supporting_evidence: List[Evidence] = Field(default_factory=list)
    referenced_entities: List[str] = Field(default_factory=list)
