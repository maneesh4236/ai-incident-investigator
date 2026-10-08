"""
Extracts entities (services, errors, components, databases, APIs, events,
systems) and relationships from incident data.

Upload-time extraction is fully deterministic (no Gemini calls):
  * `extract_from_events` - for parsed logs: entities from structured event
    fields, relationships only where the text states them, each carrying
    evidence event ids, a timestamp and a `basis`.
  * `extract_deterministic` - regex/heuristic extraction for non-log documents.

Relationship semantics:
  * CAUSES / TRIGGERS are only created from explicit causal wording inside a
    single event ("caused by", "due to", "led to", `Caused by:` chains) ->
    basis="explicit_text".
  * Same-service co-occurrence is RELATED_TO (basis="emitted_by"/"co_occurrence").
  * Temporal order is PRECEDES (basis="temporal", low confidence).
  Only CAUSES/TRIGGERS with an evidence-backed basis are ever treated as causal.

`extract()` / `_extract_with_llm` are kept for backward compatibility but are
not called by the upload pipeline.
"""
from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional

from app.core.logging_config import get_logger
from app.models.schemas import (
    Chunk,
    EntityType,
    ExtractionResult,
    GraphEntity,
    GraphRelationship,
    RelationType,
)
from app.services.ingestion.events import ERROR_LEVELS, EventGroup, LogEvent
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

# --- Event-based (log) extraction tables -------------------------------- #
_INFRA_HINTS = [
    (re.compile(r"\boracle\b", re.I), "Oracle DB", EntityType.DATABASE),
    (re.compile(r"\bpostgres(?:ql)?\b", re.I), "PostgreSQL", EntityType.DATABASE),
    (re.compile(r"\bmysql\b", re.I), "MySQL", EntityType.DATABASE),
    (re.compile(r"\bmongo(?:db)?\b", re.I), "MongoDB", EntityType.DATABASE),
    (re.compile(r"\bredis\b", re.I), "Redis", EntityType.DATABASE),
    (re.compile(r"\bkafka\b", re.I), "Kafka", EntityType.SYSTEM),
    (re.compile(r"\brabbitmq\b", re.I), "RabbitMQ", EntityType.SYSTEM),
    (re.compile(r"\bhikari(?:pool)?\b", re.I), "HikariPool", EntityType.COMPONENT),
    (re.compile(r"\bjdbc\b", re.I), "JDBC", EntityType.COMPONENT),
    (re.compile(r"connection pool|pool usage|pool_usage|\bdb connection pool\b", re.I), "DB Connection Pool", EntityType.COMPONENT),
]
_API_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-.]*)")
_DEPENDENCY_RE = re.compile(
    r"(?:downstream|dependency|upstream)\s+([\w.\-]+)\s+(?:unavailable|down|failed|timeout|timed out)"
    r"|([\w.\-]+)\s+dependency\s+(?:unavailable|down|failed)",
    re.I,
)
_MENTION_SERVICE_RE = re.compile(r"\b(?:in|from|to|for)\s+([a-z][\w\-]*-(?:service|svc|db|gateway))\b", re.I)
_EXPLICIT_CAUSE_RES = [
    # effect ... caused by / due to ... cause
    (re.compile(r"(?P<effect>[A-Za-z][\w\s\-]{2,60}?)\s+(?:was |were |is )?(?:caused by|due to)\s+(?P<cause>[A-Za-z][\w\s\-]{2,60})", re.I), RelationType.CAUSES),
    # cause ... caused / led to / resulted in ... effect
    (re.compile(r"(?P<cause>[A-Za-z][\w\s\-]{2,60}?)\s+(?:caused|led to|leads to|resulted in)\s+(?P<effect>[A-Za-z][\w\s\-]{2,60})", re.I), RelationType.CAUSES),
    (re.compile(r"(?P<cause>[A-Za-z][\w\s\-]{2,60}?)\s+triggered\s+(?P<effect>[A-Za-z][\w\s\-]{2,60})", re.I), RelationType.TRIGGERS),
]
_CAUSED_BY_LINE_RE = re.compile(r"^\s*Caused by:\s*((?:[\w$]+\.)*([\w$]+(?:Exception|Error)))")
_SIGNIFICANT_WARN_TAGS = frozenset(
    {"TIMEOUT", "POOL", "LATENCY", "LEAK", "STARVATION", "DEPENDENCY_UNAVAILABLE", "HTTP_5XX", "ERROR_RATE", "EXCEPTION", "CIRCUIT_BREAKER", "RETRY_EXHAUSTED"}
)
_PLACEHOLDER_RE = re.compile(r"<[A-Z]+>\w*|\b\w+=<N>")
_MAX_PRECEDES_EDGES = 20


class EntityExtractor:
    def __init__(self, llm_client: GeminiClient | None = None):
        self.llm_client = llm_client or GeminiClient()

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """Legacy entry point (LLM when configured). Not used during upload."""
        if self.llm_client.is_configured:
            result = self._extract_with_llm(chunk)
            if result.entities:
                return result
            logger.warning(f"LLM extraction returned nothing for chunk {chunk.id}, using heuristics")

        return self._extract_heuristic(chunk)

    def extract_deterministic(self, chunk: Chunk) -> ExtractionResult:
        """Upload-time extraction for non-log documents: never calls Gemini."""
        return self._extract_heuristic(chunk)

    # ------------------------------------------------------------------ #
    # Event-based extraction (logs)
    # ------------------------------------------------------------------ #
    def extract_from_events(
        self,
        investigation_id: str,
        groups: List[EventGroup],
        events_by_id: Dict[str, LogEvent],
        point_ids: Optional[Dict[str, str]] = None,
    ) -> ExtractionResult:
        point_ids = point_ids or {}
        entities: Dict[str, GraphEntity] = {}
        relationships: Dict[tuple, GraphRelationship] = {}
        services = {g.service for g in groups if g.service}

        def entity(name: str, etype: EntityType, group: Optional[EventGroup] = None) -> str:
            name = name.strip()
            existing = entities.get(name.lower())
            chunk_id = point_ids.get(group.id) if group else None
            if existing is None:
                entities[name.lower()] = GraphEntity(
                    id=str(uuid.uuid4()),
                    name=name,
                    type=etype,
                    investigation_id=investigation_id,
                    source_chunk_ids=[chunk_id] if chunk_id else [],
                )
            elif chunk_id and chunk_id not in existing.source_chunk_ids and len(existing.source_chunk_ids) < 20:
                existing.source_chunk_ids.append(chunk_id)
            return entities[name.lower()].name

        def relate(source: str, target: str, rtype: RelationType, basis: str, event: LogEvent, confidence: float):
            if source == target:
                return
            key = (source.lower(), target.lower(), rtype.value)
            rel = relationships.get(key)
            if rel is None:
                relationships[key] = GraphRelationship(
                    id=str(uuid.uuid4()),
                    source=source,
                    target=target,
                    type=rtype,
                    investigation_id=investigation_id,
                    confidence=confidence,
                    basis=basis,  # type: ignore[arg-type]
                    evidence_event_ids=[event.id],
                    evidence_chunk_ids=[point_ids[event.group_id]] if event.group_id in point_ids else [],
                    timestamp=event.ts_raw,
                )
            elif event.id not in rel.evidence_event_ids and len(rel.evidence_event_ids) < 10:
                rel.evidence_event_ids.append(event.id)

        def resolve_service(token: str) -> Optional[str]:
            token_l = token.lower()
            if token_l in {s.lower() for s in services}:
                return next(s for s in services if s.lower() == token_l)
            matches = [s for s in services if s.lower().startswith(token_l)]
            return matches[0] if len(matches) == 1 else None

        abnormal_firsts: List[tuple] = []  # (first event, entity name)

        for group in groups:
            first = events_by_id.get(group.first_event_id)
            if first is None:
                continue
            service_name = entity(group.service, EntityType.SERVICE, group) if group.service else None

            for pattern, name, etype in _INFRA_HINTS:
                if pattern.search(first.message):
                    infra = entity(name, etype, group)
                    if service_name:
                        relate(service_name, infra, RelationType.RELATED_TO, "co_occurrence", first, 0.3)

            api = _API_RE.search(first.message)
            if api:
                api_path = re.sub(r"/\d+", "/{id}", api.group(2))
                api_name = entity(f"{api.group(1)} {api_path}", EntityType.API, group)
                if service_name:
                    relate(service_name, api_name, RelationType.RELATED_TO, "emitted_by", first, 0.5)

            significant = group.level in ERROR_LEVELS or (group.level == "WARN" and group.tags & _SIGNIFICANT_WARN_TAGS)
            if significant:
                problem = entity(self._problem_name(group), EntityType.ERROR if group.level in ERROR_LEVELS else EntityType.EVENT, group)
                if service_name:
                    relate(service_name, problem, RelationType.RELATED_TO, "emitted_by", first, 0.6)
                abnormal_firsts.append((first, problem))

            for event_id in group.instance_ids(3):
                event = events_by_id.get(event_id)
                if event is None:
                    continue
                self._explicit_relations(event, service_name, entity, relate, resolve_service, group)

            if "RECOVERY" in group.tags and group.level not in ERROR_LEVELS and service_name:
                recovery = entity(self._problem_name(group), EntityType.EVENT, group)
                relate(recovery, service_name, RelationType.RECOVERS, "explicit_text", first, 0.6)

        # Temporal order of first abnormal occurrences: PRECEDES only (never causal).
        abnormal_firsts.sort(key=lambda item: item[0].seq)
        seen_pairs = 0
        for (prev_event, prev_name), (next_event, next_name) in zip(abnormal_firsts, abnormal_firsts[1:]):
            if seen_pairs >= _MAX_PRECEDES_EDGES:
                break
            if prev_name != next_name:
                relate(prev_name, next_name, RelationType.PRECEDES, "temporal", next_event, 0.3)
                seen_pairs += 1

        return ExtractionResult(entities=list(entities.values()), relationships=list(relationships.values()))

    def _explicit_relations(self, event: LogEvent, service_name, entity, relate, resolve_service, group) -> None:
        message = event.message
        dep = _DEPENDENCY_RE.search(message)
        if dep and service_name:
            target = resolve_service(dep.group(1) or dep.group(2) or "")
            if target and target != service_name:
                relate(service_name, target, RelationType.DEPENDS_ON, "explicit_text", event, 0.8)
                relate(target, service_name, RelationType.AFFECTS, "explicit_text", event, 0.7)

        mention = _MENTION_SERVICE_RE.search(message)
        if mention and event.is_abnormal:
            mentioned = resolve_service(mention.group(1))
            if mentioned and mentioned != service_name:
                problem = entity(self._problem_name(group), EntityType.ERROR if event.level in ERROR_LEVELS else EntityType.EVENT, group)
                relate(problem, mentioned, RelationType.RELATED_TO, "explicit_text", event, 0.6)

        for pattern, rtype in _EXPLICIT_CAUSE_RES:
            match = pattern.search(message)
            if match:
                cause = entity(_clean_phrase(match.group("cause")), EntityType.EVENT, group)
                effect = entity(_clean_phrase(match.group("effect")), EntityType.EVENT, group)
                relate(cause, effect, rtype, "explicit_text", event, 0.8)
                break

        if event.continuation_lines and event.exception_type:
            top = event.exception_type.split(".")[-1]
            for line in event.raw.split("\n")[1:]:
                caused = _CAUSED_BY_LINE_RE.match(line)
                if caused:
                    root = caused.group(2)
                    if root != top:
                        relate(entity(root, EntityType.ERROR, group), entity(top, EntityType.ERROR, group),
                               RelationType.CAUSES, "explicit_text", event, 0.85)
                    top = root

    @staticmethod
    def _problem_name(group: EventGroup) -> str:
        if group.exception_type:
            return group.exception_type.split(".")[-1]
        text = _PLACEHOLDER_RE.sub(" ", group.template)
        text = re.sub(r"\s+", " ", text).strip(" :-,=")
        text = re.sub(r"\s+(?:to|at|of|in|for|by|after|=)$", "", text, flags=re.I)
        words = text.split()
        return " ".join(words[:8]) or group.template[:60]

    # ------------------------------------------------------------------ #
    # LLM extraction (legacy, not used at upload)
    # ------------------------------------------------------------------ #
    def _extract_with_llm(self, chunk: Chunk) -> ExtractionResult:
        prompt = f"Incident text:\n---\n{chunk.text}\n---"
        data = self.llm_client.generate_json(prompt, system_instruction=_SYSTEM_PROMPT)

        entities: List[GraphEntity] = []
        for raw_entity in data.get("entities", []) or []:
            if not isinstance(raw_entity, dict):
                continue
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
        for raw_rel in data.get("relationships", []) or []:
            if not isinstance(raw_rel, dict):
                continue
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
                    confidence=_safe_float(raw_rel.get("confidence"), 0.7),
                    evidence_chunk_ids=[chunk.id],
                    basis="co_occurrence",  # uncited model output is never causal evidence
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
                    basis="explicit_text",
                )
            )

        # Chain adjacent known entities with RELATED_TO when no explicit
        # causal language was found, so the graph isn't left disconnected.
        # These are co-occurrence links only and are never treated as causal.
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
                        confidence=0.2,
                        evidence_chunk_ids=[chunk.id],
                        basis="co_occurrence",
                    )
                )

        return ExtractionResult(entities=list(entities.values()), relationships=relationships)


def _clean_phrase(text: str) -> str:
    words = re.sub(r"\s+", " ", text).strip(" .,:;-").split()
    return " ".join(words[:6]).title()


def _safe_float(value, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
