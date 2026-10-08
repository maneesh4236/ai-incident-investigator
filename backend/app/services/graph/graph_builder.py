"""
Builds the incident knowledge graph.

Maintains an in-process NetworkX graph per investigation (fast, always
available, used for path queries) while mirroring writes to Neo4j in batches
for persistence and for the frontend's Knowledge Graph Viewer.

A MultiDiGraph keyed by relation type is used so a RELATED_TO edge can never
overwrite (or be mistaken for) a CAUSES edge between the same two nodes.
Causal traversal (`upstream_causes`) follows only edges whose
`GraphRelationship.is_causal` is true: CAUSES/TRIGGERS with an evidence-backed
basis (explicit text or a validated Gemini citation).
"""
from __future__ import annotations

import threading
import uuid
from typing import Dict, List, Sequence

import networkx as nx

from app.core.logging_config import get_logger
from app.core.metrics import current_metrics
from app.models.schemas import (
    CAUSAL_RELATION_TYPES,
    EVIDENCE_BACKED_BASES,
    ClaimType,
    EntityType,
    ExtractionResult,
    GraphEntity,
    GraphRelationship,
    RelationType,
)
from app.services.graph.neo4j_service import Neo4jService

logger = get_logger("graph.graph_builder")

_CAUSAL_TYPE_VALUES = {t.value for t in CAUSAL_RELATION_TYPES}


class GraphBuilder:
    """One instance is shared across an investigation's lifecycle."""

    def __init__(self, neo4j_service: Neo4jService | None = None):
        self.neo4j = neo4j_service or Neo4jService()
        # investigation_id -> nx.MultiDiGraph
        self._graphs: Dict[str, nx.MultiDiGraph] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _graph_for(self, investigation_id: str) -> nx.MultiDiGraph:
        return self._graphs.setdefault(investigation_id, nx.MultiDiGraph())

    def _lock_for(self, investigation_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(investigation_id, threading.Lock())

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def ingest(self, investigation_id: str, extraction: ExtractionResult) -> None:
        """Backward-compatible single extraction ingest (batched to Neo4j)."""
        self.ingest_batch(investigation_id, extraction)

    def ingest_batch(self, investigation_id: str, extraction: ExtractionResult) -> None:
        with self._lock_for(investigation_id):
            graph = self._graph_for(investigation_id)
            for entity in extraction.entities:
                if entity.name in graph:
                    existing = graph.nodes[entity.name].setdefault("source_chunk_ids", [])
                    existing.extend(c for c in entity.source_chunk_ids if c not in existing)
                else:
                    graph.add_node(
                        entity.name,
                        id=entity.id,
                        type=entity.type.value,
                        source_chunk_ids=list(entity.source_chunk_ids),
                    )
            for rel in extraction.relationships:
                self._add_edge(graph, rel)
            nodes, edges = graph.number_of_nodes(), graph.number_of_edges()

        if self.neo4j.is_connected and (extraction.entities or extraction.relationships):
            self.neo4j.upsert_batch(extraction.entities, extraction.relationships)
            metrics = current_metrics()
            if metrics is not None:
                metrics.incr("graph_ops")

        logger.info(f"Graph for investigation {investigation_id} now has {nodes} nodes / {edges} edges")

    @staticmethod
    def _add_edge(graph: nx.MultiDiGraph, rel: GraphRelationship) -> None:
        for name in (rel.source, rel.target):
            if name not in graph:
                graph.add_node(name, type="EVENT", source_chunk_ids=[])
        graph.add_edge(
            rel.source,
            rel.target,
            key=rel.type.value,
            id=rel.id,
            type=rel.type.value,
            confidence=rel.confidence,
            evidence_chunk_ids=list(rel.evidence_chunk_ids),
            basis=rel.basis,
            evidence_event_ids=list(rel.evidence_event_ids),
            timestamp=rel.timestamp,
        )

    def add_llm_causal_edges(self, investigation_id: str, cause_chain: Sequence[dict]) -> List[GraphRelationship]:
        """Adds CAUSES edges between consecutive cause-chain steps that are both
        CONFIRMED/LIKELY and cite valid evidence. Anything weaker is not added."""
        allowed = {ClaimType.CONFIRMED.value, ClaimType.LIKELY.value}
        added: List[GraphRelationship] = []
        for prev, nxt in zip(cause_chain, cause_chain[1:]):
            if prev.get("type") not in allowed or nxt.get("type") not in allowed:
                continue
            if not prev.get("evidence_ids") or not nxt.get("evidence_ids"):
                continue
            both_confirmed = prev["type"] == nxt["type"] == ClaimType.CONFIRMED.value
            added.append(
                GraphRelationship(
                    id=str(uuid.uuid4()),
                    source=prev["step"][:120],
                    target=nxt["step"][:120],
                    type=RelationType.CAUSES,
                    investigation_id=investigation_id,
                    confidence=0.9 if both_confirmed else 0.6,
                    basis="llm_cited",
                    evidence_event_ids=sorted(set(prev["evidence_ids"]) | set(nxt["evidence_ids"])),
                )
            )
        if added:
            entities = [
                GraphEntity(id=str(uuid.uuid4()), name=name, type=EntityType.EVENT, investigation_id=investigation_id)
                for name in {r.source for r in added} | {r.target for r in added}
            ]
            self.ingest_batch(investigation_id, ExtractionResult(entities=entities, relationships=added))
        return added

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def export(self, investigation_id: str) -> tuple[List[GraphEntity], List[GraphRelationship]]:
        """Returns the current graph as schema objects, preferring Neo4j if connected."""
        if self.neo4j.is_connected:
            nodes, edges = self.neo4j.get_graph(investigation_id)
            if nodes:
                return self._neo4j_rows_to_schema(investigation_id, nodes, edges)

        return self._nx_graph_to_schema(investigation_id)

    def _nx_graph_to_schema(self, investigation_id: str):
        with self._lock_for(investigation_id):
            graph = self._graph_for(investigation_id)
            node_rows = list(graph.nodes(data=True))
            edge_rows = list(graph.edges(data=True))
        entities = [
            GraphEntity(
                id=data.get("id", name),
                name=name,
                type=data.get("type", "EVENT"),
                investigation_id=investigation_id,
                source_chunk_ids=data.get("source_chunk_ids", []),
            )
            for name, data in node_rows
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
                basis=data.get("basis", "co_occurrence"),
                evidence_event_ids=data.get("evidence_event_ids", []),
                timestamp=data.get("timestamp"),
            )
            for u, v, data in edge_rows
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
                basis=e.get("basis") or "co_occurrence",
                evidence_event_ids=e.get("evidence_event_ids") or [],
                timestamp=e.get("timestamp"),
            )
            for e in edges
        ]
        return entities, relationships

    def centrality_ranked_nodes(self, investigation_id: str) -> List[str]:
        """Entity names ranked by graph centrality. Display/context only - never
        used as a root-cause signal."""
        with self._lock_for(investigation_id):
            graph = nx.DiGraph(self._graph_for(investigation_id))
        if graph.number_of_nodes() == 0:
            return []
        scores = nx.pagerank(graph.reverse()) if graph.number_of_edges() > 0 else {n: 0 for n in graph.nodes}
        return [name for name, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    def causal_subgraph(self, investigation_id: str) -> nx.DiGraph:
        """Only evidence-backed CAUSES/TRIGGERS edges."""
        with self._lock_for(investigation_id):
            graph = self._graph_for(investigation_id)
            causal = nx.DiGraph()
            for u, v, data in graph.edges(data=True):
                if data.get("type") in _CAUSAL_TYPE_VALUES and data.get("basis") in EVIDENCE_BACKED_BASES:
                    causal.add_edge(u, v, **data)
        return causal

    def upstream_causes(self, investigation_id: str, entity_name: str, max_hops: int = 3) -> List[str]:
        """Walks backwards along evidence-backed CAUSES/TRIGGERS edges only.
        RELATED_TO / PRECEDES / DEPENDS_ON edges are never traversed."""
        causal = self.causal_subgraph(investigation_id)
        if entity_name not in causal:
            return []
        causes = list(nx.single_source_shortest_path_length(causal.reverse(), entity_name, cutoff=max_hops).keys())
        return [c for c in causes if c != entity_name]

    def explicit_relationships(self, investigation_id: str, limit: int = 40) -> List[GraphRelationship]:
        """Evidence-backed edges (explicit text / validated citation), for prompts."""
        _, relationships = self._nx_graph_to_schema(investigation_id)
        backed = [r for r in relationships if r.basis in EVIDENCE_BACKED_BASES and r.type != RelationType.RECOVERS]
        backed.sort(key=lambda r: (not r.is_causal, -r.confidence))
        return backed[:limit]

