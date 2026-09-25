"""
Repository layer for investigation state.

For a hackathon-scale project we back this with a process-local, thread-safe
in-memory store. The interface is deliberately narrow so it can be swapped
for a Postgres/Mongo-backed implementation later without touching services
or API routes.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

from app.models.schemas import Chunk, Investigation, InvestigationStatus, UploadedDocument


class IncidentRepository:
    def __init__(self):
        self._lock = threading.Lock()
        self._investigations: Dict[str, Investigation] = {}
        self._documents: Dict[str, UploadedDocument] = {}
        self._chunks: Dict[str, List[Chunk]] = {}  # investigation_id -> chunks

    # --- Investigations ---------------------------------------------- #
    def create_investigation(self, investigation: Investigation) -> Investigation:
        with self._lock:
            self._investigations[investigation.id] = investigation
        return investigation

    def get_investigation(self, investigation_id: str) -> Optional[Investigation]:
        return self._investigations.get(investigation_id)

    def update_status(self, investigation_id: str, status: InvestigationStatus) -> None:
        with self._lock:
            inv = self._investigations.get(investigation_id)
            if inv:
                inv.status = status

    def attach_report(self, investigation_id: str, report) -> None:
        with self._lock:
            inv = self._investigations.get(investigation_id)
            if inv:
                inv.report = report
                inv.status = InvestigationStatus.COMPLETED

    def list_investigations(self) -> List[Investigation]:
        return list(self._investigations.values())

    # --- Documents ------------------------------------------------------ #
    def add_document(self, document: UploadedDocument) -> None:
        with self._lock:
            self._documents[document.id] = document
            inv = self._investigations.get(document.investigation_id)
            if inv:
                inv.document_ids.append(document.id)

    def get_document(self, document_id: str) -> Optional[UploadedDocument]:
        return self._documents.get(document_id)

    def documents_for_investigation(self, investigation_id: str) -> List[UploadedDocument]:
        return [d for d in self._documents.values() if d.investigation_id == investigation_id]

    # --- Chunks ----------------------------------------------------------- #
    def add_chunks(self, investigation_id: str, chunks: List[Chunk]) -> None:
        with self._lock:
            self._chunks.setdefault(investigation_id, []).extend(chunks)

    def chunks_for_investigation(self, investigation_id: str) -> List[Chunk]:
        return self._chunks.get(investigation_id, [])


# Process-wide singleton so all API routes share the same in-memory state.
incident_repository = IncidentRepository()
