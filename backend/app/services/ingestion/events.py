"""
Internal (non-Pydantic) event types for the log pipeline.

Slotted dataclasses keep memory bounded for files with 100k+ events; the API
layer converts to Pydantic DTOs only for the handful of events it returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

LEVEL_RANK = {"TRACE": 0, "DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "CRITICAL": 4}
ABNORMAL_LEVELS = frozenset({"WARN", "ERROR", "CRITICAL"})
ERROR_LEVELS = frozenset({"ERROR", "CRITICAL"})


@dataclass(slots=True)
class LogEvent:
    id: str                      # "E00042" - stable, citable within an investigation
    seq: int                     # global order within the investigation
    document_id: str
    line_start: int
    line_end: int
    raw: str                     # complete original text, all lines, never altered
    message: str                 # first-line message without timestamp/level/service
    ts_raw: Optional[str] = None
    ts: Optional[datetime] = None  # parsed; carried forward for lines without one
    level: Optional[str] = None    # TRACE/DEBUG/INFO/WARN/ERROR/CRITICAL
    service: Optional[str] = None
    exception_type: Optional[str] = None
    has_stack_trace: bool = False
    continuation_lines: int = 0
    status_code: Optional[int] = None
    ids: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, float]] = None
    template_id: str = ""
    group_id: str = ""
    tags: frozenset = frozenset()

    @property
    def is_multiline(self) -> bool:
        return self.continuation_lines > 0

    @property
    def level_rank(self) -> int:
        return LEVEL_RANK.get(self.level or "", 1)

    @property
    def is_abnormal(self) -> bool:
        return self.level in ABNORMAL_LEVELS

    @property
    def ts_display(self) -> str:
        return self.ts_raw or "?"


@dataclass(slots=True)
class Burst:
    start_event_id: str
    end_event_id: str
    count: int
    start_ts: Optional[datetime]
    end_ts: Optional[datetime]


@dataclass
class EventGroup:
    """All occurrences of one template (service + level + masked message + exception).

    Every member event id is kept, so deduplication never loses an instance:
    the selector can always recover first / peak / last / per-burst /
    per-request-id occurrences from the event store.
    """

    id: str
    template: str
    service: Optional[str]
    level: Optional[str]
    exception_type: Optional[str]
    event_ids: List[str] = field(default_factory=list)
    seqs: List[int] = field(default_factory=list)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    metric_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    peak_event_id: Optional[str] = None
    bursts: List[Burst] = field(default_factory=list)
    distinct_request_ids: Dict[str, str] = field(default_factory=dict)  # request id -> first event id
    has_stack_trace: bool = False

    @property
    def count(self) -> int:
        return len(self.event_ids)

    @property
    def first_event_id(self) -> str:
        return self.event_ids[0]

    @property
    def last_event_id(self) -> str:
        return self.event_ids[-1]

    @property
    def level_rank(self) -> int:
        return LEVEL_RANK.get(self.level or "", 1)

    def instance_ids(self, max_instances: int) -> List[str]:
        """Distinct occurrences worth showing individually, in event order.

        Priority: first, peak metric, each burst start, first event per
        distinct request/trace id, last. Capped at `max_instances` (>= 1).
        """
        ordered: List[str] = []

        def add(eid: Optional[str]) -> None:
            if eid and eid not in ordered:
                ordered.append(eid)

        add(self.first_event_id)
        add(self.peak_event_id)
        for burst in self.bursts[1:]:
            add(burst.start_event_id)
        for eid in self.distinct_request_ids.values():
            add(eid)
        add(self.last_event_id)
        picked = ordered[: max(1, max_instances)]
        position = {eid: i for i, eid in enumerate(self.event_ids)}
        return sorted(picked, key=lambda e: position.get(e, 0))

    def metric_summary(self) -> str:
        parts: List[str] = []
        for name, stats in self.metric_stats.items():
            if stats["min"] == stats["max"]:
                continue
            parts.append(f"{name} {_fmt(stats['first'])}->{_fmt(stats['last'])} (peak {_fmt(stats['max'])})")
        return "; ".join(parts[:3])

    def time_span(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        return self.first_ts, self.last_ts


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"
