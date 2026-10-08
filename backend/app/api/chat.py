"""
POST /chat

Conversational investigation endpoint: exactly ONE logical Gemini call per
question (enforced by the request's metrics scope), grounded in the selected
evidence and, if available, the investigation's existing RCA report. Gemini
failures return a degraded, deterministic answer with HTTP 200.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_incident_repository, get_investigation_agent
from app.core.logging_config import get_logger
from app.core.metrics import metrics_scope
from app.models.schemas import ChatRequest

router = APIRouter(tags=["chat"])
logger = get_logger("api.chat")


@router.post("/chat")
def chat(
    request: ChatRequest,
    repo=Depends(get_incident_repository),
    agent=Depends(get_investigation_agent),
):
    investigation = repo.get_investigation(request.investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="investigation_id not found")

    if not repo.chunks_for_investigation(request.investigation_id):
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet for this investigation. Upload files first via /upload.",
        )

    with metrics_scope("chat", max_gemini_calls=1) as metrics:
        try:
            response = agent.ask(
                investigation_id=request.investigation_id,
                message=request.message,
                history=request.history,
                report=investigation.report,
            )
        except Exception:
            # Gemini failures never reach here (the agent falls back); this is an internal bug.
            logger.exception(f"Chat failed for {request.investigation_id}")
            raise HTTPException(
                status_code=500,
                detail="The chat service hit an internal error while answering. Please try again.",
            )
    response.metrics = metrics.to_dict()
    return response.model_dump()
