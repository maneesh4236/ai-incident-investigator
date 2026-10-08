"""
GET /graph/{investigation_id}

Returns the knowledge graph (nodes + edges) for the frontend's Knowledge
Graph Viewer.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_graph_builder, get_incident_repository
from app.models.schemas import KnowledgeGraphResponse

router = APIRouter(tags=["graph"])


@router.get("/graph/{investigation_id}")
def get_graph(  # sync: Neo4j reads run in the threadpool, not on the event loop
    investigation_id: str,
    repo=Depends(get_incident_repository),
    graph_builder=Depends(get_graph_builder),
):
    investigation = repo.get_investigation(investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="investigation_id not found")

    entities, relationships = graph_builder.export(investigation_id)
    response = KnowledgeGraphResponse(
        investigation_id=investigation_id, nodes=entities, edges=relationships
    )
    return response.model_dump()
