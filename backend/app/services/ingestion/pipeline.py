"""
Upload-time ingestion pipeline. Makes ZERO Gemini calls.

Logs:      parse complete events -> templates / tags / dedup groups
           -> one Qdrant point per group -> embed -> deterministic graph
Documents: line-aware chunks -> embed -> heuristic graph

Everything is staged in memory first and committed at the end (Qdrant, event
store, graph); `rollback` removes a document's data after a failure so a
failed upload leaves no partial state behind.
"""
from __future__ import annotations

from typing import Any, Dict

from app.core.logging_config import get_logger
from app.core.metrics import current_metrics
from app.models.schemas import DocumentType, ExtractionResult, UploadedDocument
from app.repositories.incident_repository import IncidentRepository
from app.services.graph.graph_builder import GraphBuilder
from app.services.ingestion.chunker import Chunker
from app.services.ingestion.document_loader import DocumentLoader
from app.services.ingestion.entity_extractor import EntityExtractor
from app.services.ingestion.event_processor import EventProcessor
from app.services.ingestion.log_loader import LogEventParser
from app.services.vector.embedder import Embedder
from app.services.vector.qdrant_service import QdrantService

logger = get_logger("ingestion.pipeline")


class IngestionError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class IngestionPipeline:
    def __init__(
        self,
        repo: IncidentRepository,
        parser: LogEventParser,
        processor: EventProcessor,
        document_loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        qdrant: QdrantService,
        extractor: EntityExtractor,
        graph_builder: GraphBuilder,
    ):
        self.repo = repo
        self.parser = parser
        self.processor = processor
        self.document_loader = document_loader
        self.chunker = chunker
        self.embedder = embedder
        self.qdrant = qdrant
        self.extractor = extractor
        self.graph_builder = graph_builder

    def ingest(self, investigation_id: str, document: UploadedDocument) -> Dict[str, Any]:
        metrics = current_metrics()

        def stage(name: str):
            return metrics.stage(name) if metrics is not None else _Null()

        current = "parse"
        try:
            events, groups = [], []
            if document.doc_type == DocumentType.LOG:
                with stage("parse"):
                    events = self.parser.parse_file(
                        document.stored_path, document.id, id_offset=self.repo.next_event_offset(investigation_id)
                    )
                current = "dedup"
                with stage("dedup"):
                    groups = self.processor.process(events, group_prefix=document.id[:6])
                stats = self.processor.stats(events, groups)
                events_by_id = {e.id: e for e in events}
                current = "points"
                chunks = self.chunker.points_from_groups(groups, events_by_id, document.id, investigation_id)
            else:
                text = self.document_loader.load(document.stored_path, document.doc_type)
                current = "chunk"
                chunks = self.chunker.split(text, document_id=document.id, investigation_id=investigation_id)
                stats = {"events": 0, "chunks": len(chunks)}
                events_by_id = {}

            current = "embed"
            with stage("embed"):
                if chunks:
                    vectors = self.embedder.embed_documents([c.text for c in chunks])
                    for chunk, vector in zip(chunks, vectors):
                        chunk.embedding = vector

            current = "extract"
            with stage("extract"):
                if events:
                    extraction = self.extractor.extract_from_events(
                        investigation_id, groups, events_by_id,
                        {c.metadata["group_id"]: c.id for c in chunks if c.metadata.get("group_id")},
                    )
                else:
                    entities, relationships = [], []
                    for chunk in chunks:
                        partial = self.extractor.extract_deterministic(chunk)
                        entities.extend(partial.entities)
                        relationships.extend(partial.relationships)
                    extraction = ExtractionResult(entities=entities, relationships=relationships)

            # ---- commit ----
            current = "qdrant"
            with stage("qdrant"):
                self.qdrant.upsert_chunks(chunks)
            current = "store"
            for chunk in chunks:
                chunk.embedding = None  # vectors live in Qdrant; keep the in-memory store small
            self.repo.add_chunks(investigation_id, chunks)
            if events:
                self.repo.add_events(investigation_id, events, groups)
            self.repo.set_parse_stats(investigation_id, document.id, stats)
            current = "graph"
            with stage("graph"):
                self.graph_builder.ingest_batch(investigation_id, extraction)

            if metrics is not None:
                metrics.incr("parsed_events", stats.get("events", 0))
                metrics.incr("multiline_events", stats.get("multiline_events", 0))
                metrics.incr("templates", stats.get("templates", 0))
                metrics.incr("groups", stats.get("groups", 0))
            stats = dict(stats)
            stats.update(points=len(chunks), entities=len(extraction.entities), relationships=len(extraction.relationships))
            logger.info(f"Ingested {document.filename}: {stats}")
            return stats
        except Exception as exc:
            logger.exception(f"Ingestion failed for {document.filename} at stage '{current}'")
            raise IngestionError(current, f"{type(exc).__name__}: {exc}") from exc

    def rollback(self, investigation_id: str, document: UploadedDocument) -> None:
        self.qdrant.delete_by_document(investigation_id, document.id)
        self.repo.remove_document_data(investigation_id, document.id)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
