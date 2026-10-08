"""
Repository layer for investigation state.

For a hackathon-scale project we back this with a process-local, thread-safe
in-memory store. The interface is deliberately narrow so it can be swapped
for a Postgres/Mongo-backed implementation later without touching services
or API routes.

Besides investigations/documents/chunks it now holds the parsed event store:
every `LogEvent` (full raw text) and every `EventGroup`, so deduplicated
groups can always be expanded back into concrete, citable event instances.
"""
from __future__ import annotations

import re
import threading
from typing import Dict, Iterable, List, Optional

from app.models.schemas import Chunk, Investigation, InvestigationStatus, UploadedDocument
from app.services.ingestion.events import EventGroup, LogEvent


class IncidentRepository:
    def __init__(self):
        self._lock = threading.RLock()
        self._investigations: Dict[str, Investigation] = {}
        self._documents: Dict[str, UploadedDocument] = {}
        self._chunks: Dict[str, List[Chunk]] = {}  # investigation_id -> chunks
        self._events: Dict[str, List[LogEvent]] = {}  # investigation_id -> events (seq order)
        self._events_by_id: Dict[str, Dict[str, LogEvent]] = {}
        self._groups: Dict[str, Dict[str, EventGroup]] = {}
        self._parse_stats: Dict[str, Dict[str, dict]] = {}  # investigation_id -> doc_id -> stats

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
                if status != InvestigationStatus.FAILED:
                    inv.error = None

    def try_mark_processing(self, investigation_id: str) -> bool:
        """Atomically moves an investigation to PROCESSING; False if it already is."""
        with self._lock:
            inv = self._investigations.get(investigation_id)
            if not inv or inv.status == InvestigationStatus.PROCESSING:
                return False
            inv.status = InvestigationStatus.PROCESSING
            inv.error = None
            return True

    def fail_investigation(self, investigation_id: str, reason: str) -> None:
        with self._lock:
            inv = self._investigations.get(investigation_id)
            if inv:
                inv.status = InvestigationStatus.FAILED
                inv.error = reason[:500]

    def ensure_not_processing(self, investigation_id: str, fallback: InvestigationStatus) -> None:
        """Safety net used in `finally` blocks: never leave PROCESSING behind."""
        with self._lock:
            inv = self._investigations.get(investigation_id)
            if inv and inv.status == InvestigationStatus.PROCESSING:
                inv.status = fallback

    def attach_report(self, investigation_id: str, report) -> None:
        with self._lock:
            inv = self._investigations.get(investigation_id)
            if inv:
                inv.report = report
                inv.status = InvestigationStatus.COMPLETED
                inv.error = None

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

    def remove_document_data(self, investigation_id: str, document_id: str) -> None:
        """Rolls back everything stored for one document (failed upload)."""
        with self._lock:
            self._documents.pop(document_id, None)
            inv = self._investigations.get(investigation_id)
            if inv and document_id in inv.document_ids:
                inv.document_ids.remove(document_id)
            if investigation_id in self._chunks:
                self._chunks[investigation_id] = [c for c in self._chunks[investigation_id] if c.document_id != document_id]
            if investigation_id in self._events:
                kept = [e for e in self._events[investigation_id] if e.document_id != document_id]
                self._events[investigation_id] = kept
                self._events_by_id[investigation_id] = {e.id: e for e in kept}
                live_groups = {e.group_id for e in kept}
                self._groups[investigation_id] = {
                    gid: g for gid, g in self._groups.get(investigation_id, {}).items() if gid in live_groups
                }
            self._parse_stats.get(investigation_id, {}).pop(document_id, None)

    # --- Chunks ----------------------------------------------------------- #
    def add_chunks(self, investigation_id: str, chunks: List[Chunk]) -> None:
        with self._lock:
            self._chunks.setdefault(investigation_id, []).extend(chunks)

    def chunks_for_investigation(self, investigation_id: str) -> List[Chunk]:
        return self._chunks.get(investigation_id, [])

    # --- Event store ------------------------------------------------------ #
    def next_event_offset(self, investigation_id: str) -> int:
        events = self._events.get(investigation_id)
        return events[-1].seq + 1 if events else 0

    def add_events(self, investigation_id: str, events: List[LogEvent], groups: List[EventGroup]) -> None:
        with self._lock:
            self._events.setdefault(investigation_id, []).extend(events)
            by_id = self._events_by_id.setdefault(investigation_id, {})
            for event in events:
                by_id[event.id] = event
            store = self._groups.setdefault(investigation_id, {})
            for group in groups:
                store[group.id] = group

    def events_for_investigation(self, investigation_id: str) -> List[LogEvent]:
        return self._events.get(investigation_id, [])

    def get_event(self, investigation_id: str, event_id: str) -> Optional[LogEvent]:
        return self._events_by_id.get(investigation_id, {}).get(event_id)

    def events_by_id(self, investigation_id: str) -> Dict[str, LogEvent]:
        return self._events_by_id.get(investigation_id, {})

    def groups_for_investigation(self, investigation_id: str) -> List[EventGroup]:
        return list(self._groups.get(investigation_id, {}).values())

    def get_group(self, investigation_id: str, group_id: str) -> Optional[EventGroup]:
        return self._groups.get(investigation_id, {}).get(group_id)

    def has_events(self, investigation_id: str) -> bool:
        return bool(self._events.get(investigation_id))

    def find_events(
        self, investigation_id: str, *, ids: Iterable[str] = (), terms: Iterable[str] = (), limit: int = 50
    ) -> List[LogEvent]:
        """Exact lookup for chat: events carrying any of `ids` or containing any `terms`."""
        wanted_ids = {i.lower() for i in ids if i}
        wanted_terms = [t.lower() for t in terms if t and len(t) > 2]
        if not wanted_ids and not wanted_terms:
            return []
        hits: List[LogEvent] = []
        for event in self._events.get(investigation_id, []):
            text = event.raw.lower()
            if any(i in text for i in wanted_ids) or (
                wanted_terms and any(re.search(rf"\b{re.escape(t)}\b", text) for t in wanted_terms)
            ):
                hits.append(event)
                if len(hits) >= limit:
                    break
        return hits

    def set_parse_stats(self, investigation_id: str, document_id: str, stats: dict) -> None:
        with self._lock:
            self._parse_stats.setdefault(investigation_id, {})[document_id] = stats

    def parse_stats(self, investigation_id: str) -> Dict[str, dict]:
        return dict(self._parse_stats.get(investigation_id, {}))


# Process-wide singleton so all API routes share the same in-memory state.
incident_repository = IncidentRepository()
