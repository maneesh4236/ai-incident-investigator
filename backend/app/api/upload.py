"""
POST /upload

Accepts one or more files for an investigation and runs the full ingestion
pipeline synchronously:

  file -> load text -> chunk -> embed -> store vectors
       -> extract entities/relationships -> build knowledge graph
"""
from __future__ import annotations

import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.dependencies import (
    get_chunker,
    get_document_loader,
    get_embedder,
    get_entity_extractor,
    get_graph_builder,
    get_incident_repository,
    get_log_loader,
    get_qdrant_service,
)
from app.core.logging_config import get_logger
from app.models.schemas import (
    Investigation,
    InvestigationStatus,
    UploadedDocument,
)

router = APIRouter(tags=["upload"])
logger = get_logger("api.upload")


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    investigation_id: str | None = Form(default=None),
    title: str | None = Form(default=None),
    repo=Depends(get_incident_repository),
    document_loader=Depends(get_document_loader),
    log_loader=Depends(get_log_loader),
    chunker=Depends(get_chunker),
    embedder=Depends(get_embedder),
    qdrant_service=Depends(get_qdrant_service),
    entity_extractor=Depends(get_entity_extractor),
    graph_builder=Depends(get_graph_builder),
):
    settings = get_settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    if not investigation_id:
        investigation_id = str(uuid.uuid4())
        repo.create_investigation(
            Investigation(
                id=investigation_id,
                title=title or f"Investigation {investigation_id[:8]}",
                status=InvestigationStatus.PROCESSING,
            )
        )
    elif not repo.get_investigation(investigation_id):
        raise HTTPException(status_code=404, detail="investigation_id not found")

    repo.update_status(investigation_id, InvestigationStatus.PROCESSING)

    uploaded_docs = []
    total_chunks = 0

    for upload_file in files:
        contents = await upload_file.read()
        if len(contents) > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{upload_file.filename} exceeds max upload size")

        doc_id = str(uuid.uuid4())
        doc_type = document_loader.infer_doc_type(upload_file.filename)
        stored_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{upload_file.filename}")
        with open(stored_path, "wb") as f:
            f.write(contents)

        document = UploadedDocument(
            id=doc_id,
            filename=upload_file.filename,
            doc_type=doc_type,
            investigation_id=investigation_id,
            stored_path=stored_path,
            size_bytes=len(contents),
        )
        repo.add_document(document)
        uploaded_docs.append(document)

        # --- Ingestion pipeline ---
        if doc_type.value == "LOG":
            lines = log_loader.load(stored_path)
            text = log_loader.to_text(lines)
        else:
            text = document_loader.load(stored_path, doc_type)

        chunks = chunker.split(text, document_id=doc_id, investigation_id=investigation_id)
        if chunks:
            embeddings = embedder.embed_documents([c.text for c in chunks])
            for chunk, vector in zip(chunks, embeddings):
                chunk.embedding = vector
            qdrant_service.upsert_chunks(chunks)
            repo.add_chunks(investigation_id, chunks)

            for chunk in chunks:
                extraction = entity_extractor.extract(chunk)
                graph_builder.ingest(investigation_id, extraction)

        total_chunks += len(chunks)
        logger.info(f"Ingested {upload_file.filename} -> {len(chunks)} chunks")

    repo.update_status(investigation_id, InvestigationStatus.PENDING)

    return {
        "investigation_id": investigation_id,
        "documents": [d.model_dump() for d in uploaded_docs],
        "total_chunks": total_chunks,
        "status": "ready_for_investigation",
    }
