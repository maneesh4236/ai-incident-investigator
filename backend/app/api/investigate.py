"""
POST /investigate           -> runs the RCA pipeline for an investigation
GET  /timeline/{id}         -> returns the reconstructed timeline
GET  /report/{id}           -> returns the generated RCA report

`/investigate` makes exactly ONE logical Gemini call (enforced by the
request's metrics scope). Gemini problems (429/5xx/timeout/quota/invalid
output) produce a degraded, deterministic report with HTTP 200; only internal
errors return a structured 500, and the investigation is never left in
PROCESSING.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.core.dependencies import get_incident_repository, get_investigation_service
from app.core.logging_config import get_logger
from app.core.metrics import metrics_scope
from app.models.schemas import InvestigateRequest, InvestigationStatus
from app.services.llm.gemini_client import outcome_category

router = APIRouter(tags=["investigate"])
logger = get_logger("api.investigate")

_DEFAULT_QUESTION = "What was the root cause of this incident and how did it unfold?"


@router.post("/investigate")
def investigate(
    request: InvestigateRequest,
    repo=Depends(get_incident_repository),
    service=Depends(get_investigation_service),
):
    investigation = repo.get_investigation(request.investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="investigation_id not found")

    if not repo.chunks_for_investigation(request.investigation_id):
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet for this investigation. Upload files first via /upload.",
        )

    if not repo.try_mark_processing(request.investigation_id):
        raise HTTPException(status_code=409, detail="This investigation is already being processed.")

    question = request.question or _DEFAULT_QUESTION
    request_id = uuid.uuid4().hex[:12]
    report = None
    with metrics_scope("investigate", max_gemini_calls=1) as metrics:
        try:
            report = service.run(request.investigation_id, question)
            repo.attach_report(request.investigation_id, report)
        except Exception as exc:
            logger.exception(f"Investigation failed for {request.investigation_id}")
            repo.fail_investigation(request.investigation_id, f"investigation failed: {type(exc).__name__}: {exc}")
            raise HTTPException(
                status_code=500,
                detail={"error": "investigation_failed", "investigation_id": request.investigation_id,
                        "reason": type(exc).__name__},
            )
        finally:
            repo.ensure_not_processing(request.investigation_id, InvestigationStatus.FAILED)

    report.metrics = metrics.to_dict()
    logger.info(
        f"investigation request_id={request_id} investigation={request.investigation_id} "
        f"reasoning_mode={'deterministic_fallback' if report.degraded else 'gemini'} degraded={report.degraded} "
        f"outcome={outcome_category(report.degradation_reason)} fallback_reason={report.degradation_reason or 'none'} "
        f"gemini_logical={metrics.gemini_logical_calls} attempts={metrics.gemini_api_attempts} "
        f"model={get_settings().GEMINI_MODEL} evidence_events={metrics.selected_events} "
        f"evidence_tokens={metrics.evidence_tokens_used} rca_type={report.root_cause.root_cause_type.value} "
        f"confidence={report.confidence:.2f} latency_ms={metrics.stage_ms.get('total')}"
    )
    return report.model_dump()


@router.get("/timeline/{investigation_id}")
def get_timeline(investigation_id: str, repo=Depends(get_incident_repository)):
    investigation = repo.get_investigation(investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="investigation_id not found")
    if not investigation.report:
        raise HTTPException(status_code=404, detail="No report yet. Call /investigate first.")
    return investigation.report.timeline.model_dump()


@router.get("/report/{investigation_id}")
def get_report(investigation_id: str, repo=Depends(get_incident_repository)):
    investigation = repo.get_investigation(investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="investigation_id not found")
    if not investigation.report:
        raise HTTPException(status_code=404, detail="No report yet. Call /investigate first.")
    return investigation.report.model_dump()


@router.get("/investigations/{investigation_id}")
def get_investigation_status(investigation_id: str, repo=Depends(get_incident_repository)):
    """Lifecycle status without fetching the report: lets the UI distinguish "report not
    generated yet" (has_report=false) from "investigation unknown" (404 - e.g. the in-memory
    store was cleared by a backend restart) without probing /report and treating 404 as normal."""
    investigation = repo.get_investigation(investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="investigation_id not found")
    return {
        "id": investigation.id,
        "title": investigation.title,
        "status": investigation.status,
        "created_at": investigation.created_at,
        "document_count": len(investigation.document_ids),
        "has_report": investigation.report is not None,
        "degraded": investigation.report.degraded if investigation.report else None,
        "error": investigation.error,
    }


@router.get("/investigations")
def list_investigations(repo=Depends(get_incident_repository)):
    return [
        {
            "id": inv.id,
            "title": inv.title,
            "status": inv.status,
            "created_at": inv.created_at,
            "document_count": len(inv.document_ids),
            "error": inv.error,
        }
        for inv in repo.list_investigations()
    ]
