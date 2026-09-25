"""
Thin repository-style wrapper around the Neo4j Python driver.

All raw Cypher lives here so the rest of the codebase never has to know
Cypher syntax — `graph_builder.py` and `graph_retriever.py` call these
methods with plain Python objects.
"""
from __future__ import annotations

from typing import List, Optional

from neo4j import GraphDatabase

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import GraphEntity, GraphRelationship

logger = get_logger("graph.neo4j_service")


class Neo4jService:
    def __init__(self):
        settings = get_settings()
        self._settings = settings
        self._driver = None
        try:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {settings.NEO4J_URI}")
        except Exception as exc:  # pragma: no cover - depends on infra availability
            logger.warning(f"Neo4j unavailable ({exc}); running in in-memory fallback mode")
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

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def get_graph(self, investigation_id: str) -> tuple[list[dict], list[dict]]:
        if not self._driver:
            return [], []
        node_query = "MATCH (e:Entity {investigation_id: $iid}) RETURN e"
        edge_query = """
        MATCH (a:Entity {investigation_id: $iid})-[r]->(b:Entity {investigation_id: $iid})
        RETURN a.name AS source, b.name AS target, type(r) AS type, r.confidence AS confidence, r.id AS id
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
