"""
Reconstructs a chronological incident timeline from retrieved chunks.

Strategy:
  1. Pull any explicit timestamps out of the chunk text/metadata.
  2. Ask the LLM to turn the (timestamp-tagged) evidence into a clean,
     ordered sequence of timeline events.
  3. If the LLM is unavailable, fall back to ordering chunks by whatever
     timestamps were found, then by chunk_index.
"""
from __future__ import annotations

import re
from typing import List, Optional

from app.core.logging_config import get_logger
from app.models.schemas import HybridRetrievalResult, Timeline, TimelineEvent
from app.services.llm.gemini_client import GeminiClient

logger = get_logger("reasoning.timeline_builder")

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\b\d{2}:\d{2}:\d{2}\b")

_SYSTEM_PROMPT = """You are an SRE assistant reconstructing an incident timeline from log/document excerpts.
Respond ONLY with strict JSON in this exact shape:
{
  "events": [
    {"timestamp": "12:01", "title": "Redis Timeout", "description": "Redis connections started timing out", "severity": "warning"}
  ]
}
Order events chronologically. severity is one of: info, warning, critical. Use only what the evidence supports."""


class TimelineBuilder:
    def __init__(self, llm_client: GeminiClient | None = None):
        self.llm_client = llm_client or GeminiClient()

    def build(self, investigation_id: str, retrieval: HybridRetrievalResult) -> Timeline:
        if self.llm_client.is_configured:
            timeline = self._build_with_llm(investigation_id, retrieval)
            if timeline.events:
                return timeline

        logger.info("Falling back to heuristic timeline ordering")
        return self._build_heuristic(investigation_id, retrieval)

    # ------------------------------------------------------------------ #
    def _build_with_llm(self, investigation_id: str, retrieval: HybridRetrievalResult) -> Timeline:
        excerpts = "\n---\n".join(
            f"[{self._find_timestamp(rc.chunk.text) or 'unknown time'}] {rc.chunk.text[:300]}"
            for rc in retrieval.chunks[:10]
        )
        data = self.llm_client.generate_json(excerpts, system_instruction=_SYSTEM_PROMPT)
        raw_events = data.get("events", [])

        events = [
            TimelineEvent(
                timestamp=e.get("timestamp"),
                order=i,
                title=e.get("title", "Event"),
                description=e.get("description", ""),
                severity=e.get("severity", "info"),
            )
            for i, e in enumerate(raw_events)
        ]
        return Timeline(investigation_id=investigation_id, events=events)

    def _build_heuristic(self, investigation_id: str, retrieval: HybridRetrievalResult) -> Timeline:
        tagged = []
        for rc in retrieval.chunks:
            ts = self._find_timestamp(rc.chunk.text)
            tagged.append((ts, rc.chunk))

        # Chunks with a real timestamp first (sorted lexicographically, which
        # works for ISO-like and HH:MM:SS formats), then untimed chunks by index.
        timed = sorted([t for t in tagged if t[0]], key=lambda t: t[0])
        untimed = [t for t in tagged if not t[0]]
        ordered = timed + untimed

        events = []
        for i, (ts, chunk) in enumerate(ordered[:12]):
            snippet = chunk.text.strip().replace("\n", " ")[:140]
            events.append(
                TimelineEvent(
                    timestamp=ts,
                    order=i,
                    title=snippet[:60] + ("..." if len(snippet) > 60 else ""),
                    description=snippet,
                    severity="warning" if any(k in snippet.lower() for k in ("error", "fail", "timeout")) else "info",
                    source_chunk_ids=[chunk.id],
                )
            )
        return Timeline(investigation_id=investigation_id, events=events)

    @staticmethod
    def _find_timestamp(text: str) -> Optional[str]:
        match = _TIMESTAMP_RE.search(text)
        return match.group(0) if match else None
