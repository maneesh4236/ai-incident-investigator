"""
Reconstructs a chronological incident timeline from complete events.

No Gemini call is made here. The timeline is built deterministically from the
selected evidence (ordered by parsed datetime, not by string), and each entry
gets a phase from transparent rules:

  1. recovery / incident-resolved wording (non-error)     -> RECOVERY
  2. alert / remediation / diagnosis or leak findings /
     monitoring sources                                  -> CONTEXT
  3. deployment or config change before the first anomaly -> PRECURSOR
  4. WARN before the first ERROR: first WARN template     -> ANOMALY
                                  later WARN templates    -> DEGRADATION
  5. ERROR/CRITICAL: other service + dependency/5xx wording, or naming the
     first failing service                                -> PROPAGATION
                     otherwise                            -> FAILURE
  6. WARN after the first ERROR                           -> DEGRADATION
  7. anything else (baseline INFO etc.)                    -> CONTEXT

The single investigation Gemini call may relabel phases, but only for event
ids already in this skeleton (`apply_phase_overrides`), so it cannot invent
timeline entries. Without Gemini this skeleton is the timeline.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import HybridRetrievalResult, Timeline, TimelineEvent, TimelinePhase
from app.services.ingestion.events import ERROR_LEVELS, EventGroup, LogEvent
from app.services.reasoning.evidence_selector import EvidencePack, SelectedItem

logger = get_logger("reasoning.timeline_builder")

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\b\d{2}:\d{2}:\d{2}\b")
_MILESTONE_TAGS = frozenset({"DIAGNOSIS", "INCIDENT_DECLARED", "REMEDIATION"})
_ALERTING_SERVICE_RE = re.compile(r"alert|monitor|pager|sre-|oncall|on-call", re.I)
_PHASE_PRIORITY = {
    TimelinePhase.FAILURE: 0,
    TimelinePhase.ANOMALY: 1,
    TimelinePhase.RECOVERY: 2,
    TimelinePhase.PROPAGATION: 3,
    TimelinePhase.PRECURSOR: 4,
    TimelinePhase.DEGRADATION: 5,
    TimelinePhase.CONTEXT: 6,
}


def _severity(level: Optional[str]) -> str:
    if level in ERROR_LEVELS:
        return "critical"
    if level == "WARN":
        return "warning"
    return "info"


def _title(message: str, limit: int = 90) -> str:
    if len(message) <= limit:
        return message
    cut = message[:limit].rsplit(" ", 1)[0]
    return cut + " ..."


class TimelineBuilder:
    def __init__(self, llm_client=None):
        # `llm_client` accepted for backward compatibility; the timeline makes no LLM call.
        self.settings = get_settings()

    # ------------------------------------------------------------------ #
    def build_skeleton(self, investigation_id: str, pack: EvidencePack, group_point_ids: Dict[str, str] | None = None) -> Timeline:
        group_point_ids = group_point_ids or {}
        items = pack.items
        if not items:
            return Timeline(investigation_id=investigation_id, events=[])

        first_abnormal = next((i.event for i in items if i.event.level in ("WARN", "ERROR", "CRITICAL")), None)
        first_error = next((i.event for i in items if i.event.level in ERROR_LEVELS), None)
        first_warn_group = next((i.group.id for i in items if i.event.level == "WARN"), None)

        # One entry per group (its first selected instance), listing every selected instance id.
        by_group: Dict[str, List[SelectedItem]] = {}
        for item in items:
            by_group.setdefault(item.group.id, []).append(item)

        entries = []
        for group_id, group_items in by_group.items():
            lead = group_items[0]
            phase = self.phase_for(lead.event, lead.group, first_abnormal, first_error, first_warn_group)
            entries.append((lead, group_items, phase))

        cap = self.settings.TIMELINE_MAX_EVENTS
        if len(entries) > cap:
            # Tier 0: first entry of every phase + each milestone (diagnosis, declaration, remediation).
            # Tier 1: first entry of every (phase, service). Tier 2: the rest by phase priority, score.
            chronological = sorted(entries, key=lambda e: pack.items.index(e[0]))
            tier0, tier1, seen0, seen1 = [], [], set(), set()
            for entry in chronological:
                phase = entry[2]
                markers = sorted(entry[0].event.tags & _MILESTONE_TAGS)
                key0 = (phase, markers[0]) if phase == TimelinePhase.CONTEXT and markers else (
                    (phase, None) if phase != TimelinePhase.CONTEXT else None
                )
                if key0 is not None and key0 not in seen0:
                    tier0.append(entry)
                    seen0.add(key0)
                    continue
                key1 = (phase, entry[0].event.service)
                if phase != TimelinePhase.CONTEXT and key1 not in seen1:
                    tier1.append(entry)
                    seen1.add(key1)
            tier1.sort(key=lambda e: (_PHASE_PRIORITY[e[2]], -e[0].score))
            rest = [e for e in entries if e not in tier0 and e not in tier1]
            rest.sort(key=lambda e: (_PHASE_PRIORITY[e[2]], -e[0].score))
            entries = (tier0 + tier1 + rest)[:cap]

        entries.sort(key=lambda e: pack.items.index(e[0]))
        timeline_events = []
        for order, (lead, group_items, phase) in enumerate(entries):
            event, group = lead.event, lead.group
            ids = [gi.event.id for gi in group_items]
            occurrences = group.count
            description = event.raw.split("\n")[0]
            if occurrences > 1:
                description += f"  (x{occurrences} occurrences of this template)"
            timeline_events.append(
                TimelineEvent(
                    timestamp=event.ts_raw,
                    order=order,
                    title=_title(event.message),
                    description=description,
                    severity=_severity(event.level),
                    source_chunk_ids=[group_point_ids[group.id]] if group.id in group_point_ids else [],
                    phase=phase,
                    event_ids=ids,
                    service=event.service,
                    occurrences=occurrences,
                )
            )
        return Timeline(investigation_id=investigation_id, events=timeline_events)

    @staticmethod
    def phase_for(
        event: LogEvent,
        group: EventGroup,
        first_abnormal: Optional[LogEvent],
        first_error: Optional[LogEvent],
        first_warn_group: Optional[str],
    ) -> TimelinePhase:
        tags = event.tags
        before_abnormal = first_abnormal is None or _before(event, first_abnormal)
        before_error = first_error is None or _before(event, first_error)
        if tags & {"RECOVERY", "INCIDENT_RESOLVED"} and event.level not in ERROR_LEVELS:
            return TimelinePhase.RECOVERY
        if tags & {"INCIDENT_DECLARED", "REMEDIATION", "DIAGNOSIS", "LEAK"} or (
            event.service and _ALERTING_SERVICE_RE.search(event.service)
        ):
            return TimelinePhase.CONTEXT
        if tags & {"DEPLOYMENT", "CONFIG_CHANGE"} and before_abnormal:
            return TimelinePhase.PRECURSOR
        if event.level == "WARN" and before_error:
            return TimelinePhase.ANOMALY if group.id == first_warn_group else TimelinePhase.DEGRADATION
        if event.level in ERROR_LEVELS:
            if first_error is not None and event.service and event.service != first_error.service:
                first_service = (first_error.service or "").lower()
                stem = first_service.split("-")[0] if first_service else ""
                mentions_first = bool(stem) and stem in event.message.lower()
                if tags & {"DEPENDENCY_UNAVAILABLE", "HTTP_5XX"} or mentions_first:
                    return TimelinePhase.PROPAGATION
            return TimelinePhase.FAILURE
        if event.level == "WARN":
            return TimelinePhase.DEGRADATION
        return TimelinePhase.CONTEXT

    @staticmethod
    def apply_phase_overrides(timeline: Timeline, overrides: Dict[str, TimelinePhase]) -> int:
        """Applies Gemini's phase labels to existing entries only. Returns count applied."""
        applied = 0
        for entry in timeline.events:
            for event_id in entry.event_ids:
                phase = overrides.get(event_id)
                if phase is not None and phase != entry.phase:
                    entry.phase = phase
                    applied += 1
                    break
        return applied

    # ------------------------------------------------------------------ #
    # Legacy path (non-log documents only): ordered by timestamp, no LLM.
    # ------------------------------------------------------------------ #
    def build(self, investigation_id: str, retrieval: HybridRetrievalResult) -> Timeline:
        return self._build_heuristic(investigation_id, retrieval)

    def _build_heuristic(self, investigation_id: str, retrieval: HybridRetrievalResult) -> Timeline:
        tagged = []
        for rc in retrieval.chunks:
            ts = self._find_timestamp(rc.chunk.text)
            tagged.append((ts, rc.chunk))

        timed = sorted([t for t in tagged if t[0]], key=lambda t: t[0])
        untimed = [t for t in tagged if not t[0]]
        ordered = timed + untimed

        events = []
        for i, (ts, chunk) in enumerate(ordered[:12]):
            first_line = chunk.text.strip().split("\n")[0]
            events.append(
                TimelineEvent(
                    timestamp=ts,
                    order=i,
                    title=_title(first_line, 60),
                    description=first_line,
                    severity="warning" if any(k in first_line.lower() for k in ("error", "fail", "timeout")) else "info",
                    source_chunk_ids=[chunk.id],
                    phase=TimelinePhase.CONTEXT,
                )
            )
        return Timeline(investigation_id=investigation_id, events=events)

    @staticmethod
    def _find_timestamp(text: str) -> Optional[str]:
        match = _TIMESTAMP_RE.search(text)
        return match.group(0) if match else None


def _before(a: LogEvent, b: LogEvent) -> bool:
    if a.document_id != b.document_id and a.ts and b.ts:
        return (a.ts, a.seq) < (b.ts, b.seq)
    return a.seq < b.seq


def skeleton_prompt_lines(timeline: Timeline) -> List[str]:
    return [f"{e.event_ids[0]} {e.phase.value if e.phase else 'CONTEXT'}" for e in timeline.events if e.event_ids]


def entries_ids(timeline: Timeline) -> Sequence[str]:
    return [eid for e in timeline.events for eid in e.event_ids]
