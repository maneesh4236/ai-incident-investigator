"""
POST /chat

Conversational investigation endpoint. Grounded in hybrid retrieval and, if
available, the investigation's existing RCA report.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_incident_repository, get_investigation_agent
from app.core.logging_config import get_logger
from app.models.schemas import ChatRequest

router = APIRouter(tags=["chat"])
logger = get_logger("api.chat")


@router.post("/chat")
async def chat(
    request: ChatRequest,
    repo=Depends(get_incident_repository),
    agent=Depends(get_investigation_agent),
):
    investigation = repo.get_investigation(request.investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="investigation_id not found")

    chunks = repo.chunks_for_investigation(request.investigation_id)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet for this investigation. Upload files first via /upload.",
        )

    response = agent.ask(
        investigation_id=request.investigation_id,
        message=request.message,
        history=request.history,
        report=investigation.report,
    )
    return response.model_dump()
