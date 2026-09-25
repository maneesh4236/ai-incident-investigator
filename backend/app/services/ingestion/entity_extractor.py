"""
Extracts entities (services, errors, components, databases, APIs, events,
systems) and relationships (CAUSES, DEPENDS_ON, RELATED_TO, CONTAINS,
TRIGGERS) from a chunk of incident text.

Two extraction strategies are combined:
  1. LLM extraction (Gemini) using a strict JSON-schema prompt — primary path.
  2. Regex/heuristic extraction — offline fallback so the pipeline still
     produces a usable (if shallower) graph without an API key, and also
     used to backstop/validate LLM output.
"""
from __future__ import annotations

import re
import uuid
from typing import List

from app.core.logging_config import get_logger
from app.models.schemas import (
    Chunk,
    EntityType,
    ExtractionResult,
    GraphEntity,
    GraphRelationship,
    RelationType,
)
from app.services.llm.gemini_client import GeminiClient

logger = get_logger("ingestion.entity_extractor")

_SYSTEM_PROMPT = """You are a Site Reliability Engineering knowledge-graph extraction engine.
Extract entities and causal/structural relationships from incident text.

Entity types: SERVICE, ERROR, COMPONENT, DATABASE, API, EVENT, SYSTEM.
Relationship types: CAUSES, DEPENDS_ON, RELATED_TO, CONTAINS, TRIGGERS.

Respond ONLY with strict JSON, no markdown, in this exact shape:
{
  "entities": [{"name": "Redis", "type": "DATABASE"}],
  "relationships": [{"source": "Redis", "target": "Timeout", "type": "CAUSES", "confidence": 0.9}]
}
Only extract what is explicitly supported by the text. Do not invent services."""

# Heuristic fallback patterns, used offline or to seed the LLM.
_KNOWN_COMPONENT_HINTS = {
    "redis": EntityType.DATABASE,
    "postgres": EntityType.DATABASE,
    "postgresql": EntityType.DATABASE,
    "mysql": EntityType.DATABASE,
    "mongodb": EntityType.DATABASE,
    "kafka": EntityType.SYSTEM,
    "rabbitmq": EntityType.SYSTEM,
    "nginx": EntityType.COMPONENT,
    "payment": EntityType.SERVICE,
    "checkout": EntityType.SERVICE,
    "auth": EntityType.SERVICE,
    "gateway": EntityType.COMPONENT,
    "api": EntityType.API,
    "queue": EntityType.COMPONENT,
    "database": EntityType.DATABASE,
    "cache": EntityType.COMPONENT,
    "load balancer": EntityType.COMPONENT,
}

_ERROR_HINTS = [
    "timeout",
    "exception",
    "failure",
    "failed",
    "error",
    "crash",
    "outage",
    "exhaustion",
    "retry storm",
    "5xx",
    "500",
    "503",
    "deadlock",
    "leak",
]

_CAUSAL_CONNECTORS = re.compile(
    r"(?P<cause>[\w\s\-]{3,40}?)\s+(?:caused|leads? to|led to|resulted in|triggered)\s+(?P<effect>[\w\s\-]{3,40})",
    re.IGNORECASE,
)


class EntityExtractor:
    def __init__(self, llm_client: GeminiClient | None = None):
        self.llm_client = llm_client or GeminiClient()

    def extract(self, chunk: Chunk) -> ExtractionResult:
        if self.llm_client.is_configured:
            result = self._extract_with_llm(chunk)
            if result.entities:
                return result
            logger.warning(f"LLM extraction returned nothing for chunk {chunk.id}, using heuristics")

        return self._extract_heuristic(chunk)

    # ------------------------------------------------------------------ #
    # LLM extraction
    # ------------------------------------------------------------------ #
    def _extract_with_llm(self, chunk: Chunk) -> ExtractionResult:
        prompt = f"Incident text:\n---\n{chunk.text}\n---"
        data = self.llm_client.generate_json(prompt, system_instruction=_SYSTEM_PROMPT)

        entities: List[GraphEntity] = []
        for raw_entity in data.get("entities", []):
            name = str(raw_entity.get("name", "")).strip()
            etype = str(raw_entity.get("type", "")).upper()
            if not name or etype not in EntityType.__members__:
                continue
            entities.append(
                GraphEntity(
                    id=str(uuid.uuid4()),
                    name=name,
                    type=EntityType[etype],
                    investigation_id=chunk.investigation_id,
                    source_chunk_ids=[chunk.id],
                )
            )

        relationships: List[GraphRelationship] = []
        for raw_rel in data.get("relationships", []):
            rtype = str(raw_rel.get("type", "")).upper()
            source = str(raw_rel.get("source", "")).strip()
            target = str(raw_rel.get("target", "")).strip()
            if not source or not target or rtype not in RelationType.__members__:
                continue
            relationships.append(
                GraphRelationship(
                    id=str(uuid.uuid4()),
                    source=source,
                    target=target,
                    type=RelationType[rtype],
                    investigation_id=chunk.investigation_id,
                    confidence=float(raw_rel.get("confidence", 0.7)),
                    evidence_chunk_ids=[chunk.id],
                )
            )

        return ExtractionResult(entities=entities, relationships=relationships)

    # ------------------------------------------------------------------ #
    # Heuristic (offline) extraction
    # ------------------------------------------------------------------ #
    def _extract_heuristic(self, chunk: Chunk) -> ExtractionResult:
        text_lower = chunk.text.lower()
        entities: dict[str, GraphEntity] = {}

        def add_entity(name: str, etype: EntityType) -> str:
            key = name.strip().title()
            if key not in entities:
                entities[key] = GraphEntity(
                    id=str(uuid.uuid4()),
                    name=key,
                    type=etype,
                    investigation_id=chunk.investigation_id,
                    source_chunk_ids=[chunk.id],
                )
            return key

        for keyword, etype in _KNOWN_COMPONENT_HINTS.items():
            if keyword in text_lower:
                add_entity(keyword, etype)

        for keyword in _ERROR_HINTS:
            if keyword in text_lower:
                add_entity(keyword, EntityType.ERROR)

        relationships: List[GraphRelationship] = []
        for match in _CAUSAL_CONNECTORS.finditer(chunk.text):
            cause = match.group("cause").strip().title()
            effect = match.group("effect").strip().title()
            if len(cause) < 3 or len(effect) < 3:
                continue
            add_entity(cause, EntityType.EVENT)
            add_entity(effect, EntityType.EVENT)
            relationships.append(
                GraphRelationship(
                    id=str(uuid.uuid4()),
                    source=cause,
                    target=effect,
                    type=RelationType.CAUSES,
                    investigation_id=chunk.investigation_id,
                    confidence=0.55,
                    evidence_chunk_ids=[chunk.id],
                )
            )

        # Chain adjacent known entities with RELATED_TO when no explicit
        # causal language was found, so the graph isn't left disconnected.
        names = list(entities.keys())
        if not relationships and len(names) >= 2:
            for a, b in zip(names, names[1:]):
                relationships.append(
                    GraphRelationship(
                        id=str(uuid.uuid4()),
                        source=a,
                        target=b,
                        type=RelationType.RELATED_TO,
                        investigation_id=chunk.investigation_id,
                        confidence=0.4,
                        evidence_chunk_ids=[chunk.id],
                    )
                )

        return ExtractionResult(entities=list(entities.values()), relationships=relationships)
