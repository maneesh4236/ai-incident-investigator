"""
Builds the incident knowledge graph.

Maintains an in-process NetworkX graph per investigation (fast, always
available, used for quick path/centrality queries) while mirroring writes to
Neo4j for persistence and for the frontend's Knowledge Graph Viewer.
"""
from __future__ import annotations

from typing import Dict, List

import networkx as nx

from app.core.logging_config import get_logger
from app.models.schemas import ExtractionResult, GraphEntity, GraphRelationship
from app.services.graph.neo4j_service import Neo4jService

logger = get_logger("graph.graph_builder")


class GraphBuilder:
    """One instance is shared across an investigation's lifecycle."""

    def __init__(self, neo4j_service: Neo4jService | None = None):
        self.neo4j = neo4j_service or Neo4jService()
        # investigation_id -> nx.DiGraph
        self._graphs: Dict[str, nx.DiGraph] = {}

    def _graph_for(self, investigation_id: str) -> nx.DiGraph:
        return self._graphs.setdefault(investigation_id, nx.DiGraph())

    def ingest(self, investigation_id: str, extraction: ExtractionResult) -> None:
        graph = self._graph_for(investigation_id)

        for entity in extraction.entities:
            graph.add_node(
                entity.name,
                id=entity.id,
                type=entity.type.value,
                source_chunk_ids=entity.source_chunk_ids,
            )
            self.neo4j.upsert_entity(entity)

        for rel in extraction.relationships:
            if rel.source not in graph:
                graph.add_node(rel.source, type="EVENT", source_chunk_ids=[])
            if rel.target not in graph:
                graph.add_node(rel.target, type="EVENT", source_chunk_ids=[])
            graph.add_edge(
                rel.source,
                rel.target,
                id=rel.id,
                type=rel.type.value,
                confidence=rel.confidence,
                evidence_chunk_ids=rel.evidence_chunk_ids,
            )
            self.neo4j.upsert_relationship(rel)

        logger.info(
            f"Graph for investigation {investigation_id} now has "
            f"{graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges"
        )

    def export(self, investigation_id: str) -> tuple[List[GraphEntity], List[GraphRelationship]]:
        """Returns the current graph as schema objects, preferring Neo4j if connected."""
        if self.neo4j.is_connected:
            nodes, edges = self.neo4j.get_graph(investigation_id)
            if nodes:
                return self._neo4j_rows_to_schema(investigation_id, nodes, edges)

        return self._nx_graph_to_schema(investigation_id)

    def _nx_graph_to_schema(self, investigation_id: str):
        graph = self._graph_for(investigation_id)
        entities = [
            GraphEntity(
                id=data.get("id", name),
                name=name,
                type=data.get("type", "EVENT"),
                investigation_id=investigation_id,
                source_chunk_ids=data.get("source_chunk_ids", []),
            )
            for name, data in graph.nodes(data=True)
        ]
        relationships = [
            GraphRelationship(
                id=data.get("id", f"{u}->{v}"),
                source=u,
                target=v,
                type=data.get("type", "RELATED_TO"),
                investigation_id=investigation_id,
                confidence=data.get("confidence", 0.5),
                evidence_chunk_ids=data.get("evidence_chunk_ids", []),
            )
            for u, v, data in graph.edges(data=True)
        ]
        return entities, relationships

    @staticmethod
    def _neo4j_rows_to_schema(investigation_id: str, nodes: list[dict], edges: list[dict]):
        entities = [
            GraphEntity(
                id=n.get("id", n["name"]),
                name=n["name"],
                type=n.get("type", "EVENT"),
                investigation_id=investigation_id,
                source_chunk_ids=n.get("source_chunk_ids", []) or [],
            )
            for n in nodes
        ]
        relationships = [
            GraphRelationship(
                id=e.get("id") or f"{e['source']}->{e['target']}",
                source=e["source"],
                target=e["target"],
                type=e["type"],
                investigation_id=investigation_id,
                confidence=e.get("confidence") or 0.5,
            )
            for e in edges
        ]
        return entities, relationships

    def centrality_ranked_nodes(self, investigation_id: str) -> List[str]:
        """Returns entity names ranked by how 'central' they are to the graph —
        used to help surface likely root-cause candidates."""
        graph = self._graph_for(investigation_id)
        if graph.number_of_nodes() == 0:
            return []
        scores = nx.pagerank(graph.reverse()) if graph.number_of_edges() > 0 else {n: 0 for n in graph.nodes}
        return [name for name, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    def upstream_causes(self, investigation_id: str, entity_name: str, max_hops: int = 3) -> List[str]:
        """Walks backwards along CAUSES/TRIGGERS edges to find upstream causes of a symptom."""
        graph = self._graph_for(investigation_id)
        if entity_name not in graph:
            return []
        reversed_graph = graph.reverse()
        causes = list(
            nx.single_source_shortest_path_length(reversed_graph, entity_name, cutoff=max_hops).keys()
        )
        return [c for c in causes if c != entity_name]
