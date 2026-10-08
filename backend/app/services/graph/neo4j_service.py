"""
Thin repository-style wrapper around the Neo4j Python driver.

All raw Cypher lives here so the rest of the codebase never has to know
Cypher syntax — `graph_builder.py` and `graph_retriever.py` call these
methods with plain Python objects.
"""
from __future__ import annotations

import re
from typing import List, Optional

from neo4j import GraphDatabase

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import EntityType, GraphEntity, GraphRelationship, RelationType

logger = get_logger("graph.neo4j_service")


class Neo4jService:
    def __init__(self):
        settings = get_settings()
        self._settings = settings
        self._driver = None
        self.unavailable_reason: Optional[str] = None
        target = _safe_target(settings.NEO4J_URI)
        try:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                connection_timeout=5.0,  # never stall startup on an unreachable graph
            )
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {target}")
        except Exception as exc:  # pragma: no cover - depends on infra availability
            self.unavailable_reason = _classify_neo4j_error(exc)
            logger.warning(
                f"Neo4j unavailable at {target}: {self.unavailable_reason}. The knowledge graph runs in "
                "in-memory fallback mode (investigation and chat are unaffected)."
            )
            self._driver = None

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert_entity(self, entity: GraphEntity) -> None:
        if not self._driver:
            return
        query = """
        MERGE (e:Entity {name: $name, investigation_id: $investigation_id})
        ON CREATE SET e.id = $id, e.type = $type, e.source_chunk_ids = $source_chunk_ids
        ON MATCH SET e.source_chunk_ids = coalesce(e.source_chunk_ids, []) + $source_chunk_ids
        SET e:%s
        """ % entity.type.value
        with self._driver.session(database=self._settings.NEO4J_DATABASE) as session:
            session.run(
                query,
                name=entity.name,
                investigation_id=entity.investigation_id,
                id=entity.id,
                type=entity.type.value,
                source_chunk_ids=entity.source_chunk_ids,
            )

    def upsert_relationship(self, rel: GraphRelationship) -> None:
        if not self._driver:
            return
        query = """
        MATCH (a:Entity {name: $source, investigation_id: $investigation_id})
        MATCH (b:Entity {name: $target, investigation_id: $investigation_id})
        MERGE (a)-[r:%s {investigation_id: $investigation_id}]->(b)
        ON CREATE SET r.id = $id, r.confidence = $confidence, r.evidence_chunk_ids = $evidence
        """ % rel.type.value
        with self._driver.session(database=self._settings.NEO4J_DATABASE) as session:
            session.run(
                query,
                source=rel.source,
                target=rel.target,
                investigation_id=rel.investigation_id,
                id=rel.id,
                confidence=rel.confidence,
                evidence=rel.evidence_chunk_ids,
            )

    def upsert_batch(self, entities: List[GraphEntity], relationships: List[GraphRelationship]) -> int:
        """Writes entities and relationships in one session / transaction using
        UNWIND, one query per entity type and per relation type (labels and
        relationship types come from enums, so interpolation is safe).
        Returns the number of queries executed."""
        if not self._driver or (not entities and not relationships):
            return 0
        by_type: dict[str, list[dict]] = {}
        for e in entities:
            by_type.setdefault(EntityType(e.type).value, []).append(
                {"name": e.name, "investigation_id": e.investigation_id, "id": e.id,
                 "type": EntityType(e.type).value, "source_chunk_ids": e.source_chunk_ids}
            )
        rels_by_type: dict[str, list[dict]] = {}
        for r in relationships:
            rels_by_type.setdefault(RelationType(r.type).value, []).append(
                {"source": r.source, "target": r.target, "investigation_id": r.investigation_id, "id": r.id,
                 "confidence": r.confidence, "evidence": r.evidence_chunk_ids, "basis": r.basis,
                 "evidence_event_ids": r.evidence_event_ids, "timestamp": r.timestamp}
            )
        queries = 0

        def work(tx):
            nonlocal queries
            for label, rows in by_type.items():
                tx.run(
                    """
                    UNWIND $rows AS row
                    MERGE (e:Entity {name: row.name, investigation_id: row.investigation_id})
                    ON CREATE SET e.id = row.id, e.type = row.type, e.source_chunk_ids = row.source_chunk_ids
                    ON MATCH SET e.source_chunk_ids = coalesce(e.source_chunk_ids, []) + row.source_chunk_ids
                    SET e:%s
                    """ % label,
                    rows=rows,
                )
                queries += 1
            for rtype, rows in rels_by_type.items():
                tx.run(
                    """
                    UNWIND $rows AS row
                    MERGE (a:Entity {name: row.source, investigation_id: row.investigation_id})
                    MERGE (b:Entity {name: row.target, investigation_id: row.investigation_id})
                    MERGE (a)-[r:%s {investigation_id: row.investigation_id}]->(b)
                    ON CREATE SET r.id = row.id, r.confidence = row.confidence, r.evidence_chunk_ids = row.evidence,
                                  r.basis = row.basis, r.evidence_event_ids = row.evidence_event_ids,
                                  r.timestamp = row.timestamp
                    """ % rtype,
                    rows=rows,
                )
                queries += 1

        try:
            with self._driver.session(database=self._settings.NEO4J_DATABASE) as session:
                session.execute_write(work)
        except Exception as exc:  # graph persistence must never break ingestion
            logger.warning(f"Neo4j batch upsert failed ({exc}); in-memory graph remains authoritative")
        return queries

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def get_graph(self, investigation_id: str) -> tuple[list[dict], list[dict]]:
        if not self._driver:
            return [], []
        node_query = "MATCH (e:Entity {investigation_id: $iid}) RETURN e"
        edge_query = """
        MATCH (a:Entity {investigation_id: $iid})-[r]->(b:Entity {investigation_id: $iid})
        RETURN a.name AS source, b.name AS target, type(r) AS type, r.confidence AS confidence, r.id AS id,
               r.basis AS basis, r.evidence_event_ids AS evidence_event_ids, r.timestamp AS timestamp
        """
        with self._driver.session(database=self._settings.NEO4J_DATABASE) as session:
            nodes = [dict(record["e"]) for record in session.run(node_query, iid=investigation_id)]
            edges = [dict(record) for record in session.run(edge_query, iid=investigation_id)]
        return nodes, edges

    def find_paths_from(self, entity_name: str, investigation_id: str, max_hops: int = 2) -> List[List[str]]:
        if not self._driver:
            return []
        query = f"""
        MATCH path = (start:Entity {{name: $name, investigation_id: $iid}})-[*1..{max_hops}]->(end:Entity)
        RETURN [n IN nodes(path) | n.name] AS names
        LIMIT 25
        """
        with self._driver.session(database=self._settings.NEO4J_DATABASE) as session:
            return [record["names"] for record in session.run(query, name=entity_name, iid=investigation_id)]

    def find_entities_by_keyword(self, keyword: str, investigation_id: str) -> List[dict]:
        if not self._driver:
            return []
        query = """
        MATCH (e:Entity {investigation_id: $iid})
        WHERE toLower(e.name) CONTAINS toLower($keyword)
        RETURN e
        """
        with self._driver.session(database=self._settings.NEO4J_DATABASE) as session:
            return [dict(record["e"]) for record in session.run(query, iid=investigation_id, keyword=keyword)]


def _safe_target(uri: str) -> str:
    """scheme://host:port only - never credentials or query parameters."""
    match = re.match(r"^\s*([A-Za-z0-9+]+)://(?:[^@/]*@)?([^/?#\s:]+)(?::(\d+))?", uri or "")
    if not match:
        return "(invalid NEO4J_URI)"
    scheme, host, port = match.groups()
    return f"{scheme}://{host}:{port or 7687}"


def _classify_neo4j_error(exc: Exception) -> str:
    """Human-readable, credential-free reason for a failed Neo4j connection."""
    name = type(exc).__name__
    text = str(exc).lower()
    if "resolve" in text or "getaddrinfo" in text or "name or service not known" in text or "nodename" in text:
        return (f"DNS resolution failed for the host ({name}) - the Aura instance may have been deleted/paused or "
                "the hostname is wrong")
    if "auth" in name.lower() or "unauthorized" in text or "authentication" in text:
        return f"authentication failed ({name}) - check NEO4J_USER / NEO4J_PASSWORD"
    if "refused" in text or "timed out" in text or "timeout" in text or "unavailable" in name.lower():
        return f"server not reachable ({name}) - instance stopped, or network/firewall blocks port 7687"
    if "scheme" in text or "uri" in text or "configuration" in name.lower():
        return f"invalid NEO4J_URI ({name})"
    return name
