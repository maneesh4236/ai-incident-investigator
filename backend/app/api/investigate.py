"""
POST /investigate           -> runs the RCA pipeline for an investigation
GET  /timeline/{id}         -> returns the reconstructed timeline
GET  /report/{id}           -> returns the generated RCA report
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import (
    get_hybrid_retriever,
    get_incident_repository,
    get_report_generator,
    get_root_cause_analyzer,
    get_timeline_builder,
)
from app.core.logging_config import get_logger
from app.models.schemas import InvestigateRequest, InvestigationStatus

router = APIRouter(tags=["investigate"])
logger = get_logger("api.investigate")

_DEFAULT_QUESTION = "What was the root cause of this incident and how did it unfold?"


@router.post("/investigate")
async def investigate(
    request: InvestigateRequest,
    repo=Depends(get_incident_repository),
    hybrid_retriever=Depends(get_hybrid_retriever),
    root_cause_analyzer=Depends(get_root_cause_analyzer),
    timeline_builder=Depends(get_timeline_builder),
    report_generator=Depends(get_report_generator),
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

    repo.update_status(request.investigation_id, InvestigationStatus.PROCESSING)
    question = request.question or _DEFAULT_QUESTION

    try:
        retrieval = hybrid_retriever.retrieve(request.investigation_id, question)

        document_names = {
            d.id: d.filename for d in repo.documents_for_investigation(request.investigation_id)
        }
        root_cause = root_cause_analyzer.analyze(request.investigation_id, retrieval, document_names)
        timeline = timeline_builder.build(request.investigation_id, retrieval)
        report = report_generator.generate(request.investigation_id, root_cause, timeline)

        repo.attach_report(request.investigation_id, report)
        return report.model_dump()
    except Exception:
        repo.update_status(request.investigation_id, InvestigationStatus.FAILED)
        logger.exception(f"Investigation failed for {request.investigation_id}")
        raise HTTPException(status_code=500, detail="Investigation pipeline failed. See server logs.")


@router.get("/timeline/{investigation_id}")
async def get_timeline(investigation_id: str, repo=Depends(get_incident_repository)):
    investigation = repo.get_investigation(investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="investigation_id not found")
    if not investigation.report:
        raise HTTPException(status_code=404, detail="No report yet. Call /investigate first.")
    return investigation.report.timeline.model_dump()


@router.get("/report/{investigation_id}")
async def get_report(investigation_id: str, repo=Depends(get_incident_repository)):
    investigation = repo.get_investigation(investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="investigation_id not found")
    if not investigation.report:
        raise HTTPException(status_code=404, detail="No report yet. Call /investigate first.")
    return investigation.report.model_dump()


@router.get("/investigations")
async def list_investigations(repo=Depends(get_incident_repository)):
    return [
        {
            "id": inv.id,
            "title": inv.title,
            "status": inv.status,
            "created_at": inv.created_at,
            "document_count": len(inv.document_ids),
        }
        for inv in repo.list_investigations()
    ]
