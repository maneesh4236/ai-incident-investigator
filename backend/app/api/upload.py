"""
POST /upload

Accepts one or more files for an investigation and runs the deterministic
ingestion pipeline (no Gemini calls):

  logs:      file -> complete events -> templates/dedup -> group points
             -> embed -> Qdrant -> event store -> knowledge graph
  documents: file -> line-aware chunks -> embed -> Qdrant -> graph

A plain `def` route: FastAPI runs it in its threadpool, so parsing and
embedding never block the event loop. The request runs inside a metrics scope
that allows 0 Gemini calls, and the investigation is never left PROCESSING.
"""
from __future__ import annotations

import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.dependencies import get_document_loader, get_incident_repository, get_ingestion_pipeline
from app.core.logging_config import get_logger
from app.core.metrics import metrics_scope
from app.models.schemas import Investigation, InvestigationStatus, UploadedDocument
from app.services.ingestion.pipeline import IngestionError

router = APIRouter(tags=["upload"])
logger = get_logger("api.upload")

_COPY_CHUNK = 1024 * 1024


@router.post("/upload")
def upload_documents(
    files: List[UploadFile] = File(...),
    investigation_id: str | None = Form(default=None),
    title: str | None = Form(default=None),
    repo=Depends(get_incident_repository),
    document_loader=Depends(get_document_loader),
    pipeline=Depends(get_ingestion_pipeline),
):
    settings = get_settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    if not investigation_id:
        investigation_id = str(uuid.uuid4())
        repo.create_investigation(
            Investigation(
                id=investigation_id,
                title=title or f"Investigation {investigation_id[:8]}",
                status=InvestigationStatus.PENDING,
            )
        )
    elif not repo.get_investigation(investigation_id):
        raise HTTPException(status_code=404, detail="investigation_id not found")

    if not repo.try_mark_processing(investigation_id):
        raise HTTPException(status_code=409, detail="This investigation is already being processed.")

    uploaded_docs: List[UploadedDocument] = []
    doc_stats = []
    total_points = 0
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    with metrics_scope("upload", max_gemini_calls=0) as metrics:
        try:
            for upload_file in files:
                doc_id = str(uuid.uuid4())
                safe_name = os.path.basename(upload_file.filename or "upload")
                stored_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{safe_name}")
                size = _stream_to_disk(upload_file, stored_path, max_bytes)
                if size is None:
                    raise HTTPException(status_code=413, detail=f"{safe_name} exceeds max upload size")

                doc_type = document_loader.refine_doc_type(document_loader.infer_doc_type(safe_name), stored_path)
                document = UploadedDocument(
                    id=doc_id,
                    filename=safe_name,
                    doc_type=doc_type,
                    investigation_id=investigation_id,
                    stored_path=stored_path,
                    size_bytes=size,
                )
                repo.add_document(document)
                try:
                    stats = pipeline.ingest(investigation_id, document)
                except IngestionError:
                    pipeline.rollback(investigation_id, document)
                    raise
                uploaded_docs.append(document)
                doc_stats.append({"document_id": doc_id, "filename": safe_name, **stats})
                total_points += stats.get("points", 0)

            repo.update_status(investigation_id, InvestigationStatus.PENDING)
        except HTTPException as exc:
            repo.fail_investigation(investigation_id, f"upload rejected: {exc.detail}")
            raise
        except IngestionError as exc:
            repo.fail_investigation(investigation_id, f"ingestion failed at {exc.stage}: {exc}")
            raise HTTPException(
                status_code=500,
                detail={"error": "ingestion_failed", "stage": exc.stage, "investigation_id": investigation_id},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected upload failure")
            repo.fail_investigation(investigation_id, f"unexpected upload failure: {type(exc).__name__}")
            raise HTTPException(
                status_code=500,
                detail={"error": "ingestion_failed", "stage": "unexpected", "investigation_id": investigation_id},
            )
        finally:
            repo.ensure_not_processing(investigation_id, InvestigationStatus.FAILED)

    return {
        "investigation_id": investigation_id,
        "documents": [d.model_dump() for d in uploaded_docs],
        "total_chunks": total_points,
        "status": "ready_for_investigation",
        "parse_stats": doc_stats,
        "metrics": metrics.to_dict(),
    }


def _stream_to_disk(upload_file: UploadFile, path: str, max_bytes: int) -> int | None:
    """Copies the upload to disk in 1 MB pieces; returns size, or None (and deletes) if too large."""
    size = 0
    with open(path, "wb") as out:
        while True:
            piece = upload_file.file.read(_COPY_CHUNK)
            if not piece:
                break
            size += len(piece)
            if size > max_bytes:
                break
            out.write(piece)
    if size > max_bytes:
        os.remove(path)
        return None
    return size

