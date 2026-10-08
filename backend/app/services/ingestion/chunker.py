"""
Builds the units that are embedded and stored in Qdrant.

* Logs: one point per deduplicated event group (`points_from_groups`). The
  point text is the group's representative event (whole lines only), sized to
  fit the embedder's 256 word-piece window, so retrieval sees the full point
  and the number of points scales with templates, not with lines. The full
  raw text of every event stays in the event store.
* Non-log documents: line-aware windows (`split`). Whole lines are kept and
  newlines preserved; only a single line longer than the window is split by
  words.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import Chunk
from app.services.ingestion.events import EventGroup, LogEvent

logger = get_logger("ingestion.chunker")

_POINT_EVENT_IDS_CAP = 50


class Chunker:
    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
        self.overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP_TOKENS
        self.point_max_chars = settings.QDRANT_POINT_MAX_CHARS

    # ------------------------------------------------------------------ #
    # Non-log documents
    # ------------------------------------------------------------------ #
    def split(self, text: str, document_id: str, investigation_id: str) -> List[Chunk]:
        if not text.split():
            return []

        chunks: List[Chunk] = []
        current: List[str] = []
        current_words = 0

        def emit(lines: List[str]) -> None:
            chunk_text = "\n".join(lines).strip()
            if chunk_text:
                chunks.append(self._chunk(chunk_text, document_id, investigation_id, len(chunks)))

        for line in text.splitlines():
            words = len(line.split())
            if words == 0:
                continue
            if words > self.chunk_size:
                if current:
                    emit(current)
                    current, current_words = [], 0
                for piece in self._split_words(line.split()):
                    emit([piece])
                continue
            if current and current_words + words > self.chunk_size:
                emit(current)
                current, current_words = self._overlap_tail(current, words)
            current.append(line)
            current_words += words

        if current:
            emit(current)
        logger.info(f"Chunked document {document_id} into {len(chunks)} chunks")
        return chunks

    def _overlap_tail(self, lines: List[str], next_words: int) -> tuple[List[str], int]:
        tail: List[str] = []
        total = 0
        for line in reversed(lines):
            words = len(line.split())
            if total + words > self.overlap:
                break
            tail.insert(0, line)
            total += words
        if total + next_words > self.chunk_size:
            return [], 0
        return tail, total

    def _split_words(self, words: List[str]) -> List[str]:
        step = max(self.chunk_size - self.overlap, 1)
        pieces = []
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if window:
                pieces.append(" ".join(window))
            if start + self.chunk_size >= len(words):
                break
        return pieces

    # ------------------------------------------------------------------ #
    # Logs
    # ------------------------------------------------------------------ #
    def points_from_groups(
        self,
        groups: List[EventGroup],
        events_by_id: Dict[str, LogEvent],
        document_id: str,
        investigation_id: str,
    ) -> List[Chunk]:
        points: List[Chunk] = []
        for group in groups:
            rep = events_by_id[group.first_event_id]
            text = self.compact_event_text(rep.raw, self.point_max_chars)
            if group.count > 1:
                span = _span(group)
                text += f"\n(x{group.count} occurrences{span})"
            chunk = self._chunk(text, document_id, investigation_id, len(points))
            chunk.metadata = {
                "kind": "event_group",
                "group_id": group.id,
                "event_ids": group.event_ids[:_POINT_EVENT_IDS_CAP],
                "count": group.count,
                "first_ts": group.first_ts.isoformat(sep=" ") if group.first_ts else None,
                "last_ts": group.last_ts.isoformat(sep=" ") if group.last_ts else None,
                "level": group.level,
                "service": group.service,
                "tags": sorted(group.tags),
            }
            points.append(chunk)
        logger.info(f"Built {len(points)} event-group points for document {document_id}")
        return points

    @staticmethod
    def compact_event_text(raw: str, max_chars: int, max_frames: Optional[int] = None) -> str:
        """Whole-line compaction of one event, used only for embedding text.

        Keeps the header line, exception / `Caused by` lines and the top stack
        frames; never cuts a line. Omitted lines are reported explicitly.
        """
        if len(raw) <= max_chars:
            return raw
        lines = raw.split("\n")
        kept = {0}
        size = len(lines[0])
        frames_kept = 0
        frame_cap = max_frames if max_frames is not None else get_settings().MAX_STACK_FRAMES_KEPT

        def is_priority(line: str) -> bool:
            return "Exception" in line or "Error" in line or line.lstrip().startswith("Caused by")

        order = [i for i in range(1, len(lines)) if is_priority(lines[i])]
        order += [i for i in range(1, len(lines)) if not is_priority(lines[i])]
        for i in order:
            line = lines[i]
            is_frame = line.lstrip().startswith("at ")
            if is_frame and frames_kept >= frame_cap:
                continue
            if size + len(line) + 1 > max_chars:
                continue
            kept.add(i)
            size += len(line) + 1
            frames_kept += int(is_frame)
        omitted = len(lines) - len(kept)
        ordered = [lines[i] for i in sorted(kept)]
        if omitted:
            ordered.append(f"[{omitted} more lines omitted]")
        return "\n".join(ordered)

    @staticmethod
    def _chunk(text: str, document_id: str, investigation_id: str, index: int) -> Chunk:
        return Chunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            investigation_id=investigation_id,
            text=text,
            chunk_index=index,
        )


def _span(group: EventGroup) -> str:
    if group.first_ts and group.last_ts and group.first_ts != group.last_ts:
        return f", {group.first_ts.time()}->{group.last_ts.time()}"
    return ""
