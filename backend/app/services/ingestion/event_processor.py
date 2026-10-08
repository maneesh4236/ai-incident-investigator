"""
Deterministic template extraction, signal tagging and deduplication.

* Templates: a lightweight Drain-style masking of variable tokens (ids,
  numbers, durations, percentages, hex, IPs, UUIDs). No new dependency.
* Tags: a documented regex table of RCA-relevant signals (timeouts, 5xx,
  pool, leak, recovery, diagnosis, ...), used for scoring and phases.
* Groups: events sharing (service, level, template, exception) are grouped.
  Groups keep EVERY member event id plus bursts, metric trends, peak event
  and distinct request ids, so no causal occurrence is lost - the evidence
  selector re-expands groups into real event instances.
"""
from __future__ import annotations

import hashlib
import re
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from app.services.ingestion.events import Burst, EventGroup, LogEvent

# Gap that separates two bursts of the same template (distinct occurrences).
BURST_GAP = timedelta(seconds=10)
MAX_DISTINCT_REQUEST_IDS = 50

_MASKS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?"), "<TS>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b|\b(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,}\b"), "<HEX>"),
    (re.compile(r"\b(?:TXN|ORD|SKU|CUST|REQ|req|txn|ord|cust)[-_]?\d+\b"), "<ID>"),
    (re.compile(r"\b([A-Za-z_][\w.]*)=\d+(?:\.\d+)?%?"), r"\1=<N>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*(ms|s|%|MB|KB|GB)(?![A-Za-z])"), r"<N>\1"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<N>"),
]

# Signal tag table (case-insensitive). Documented here so scoring/phases are auditable.
TAG_PATTERNS: Dict[str, re.Pattern] = {
    "TIMEOUT": re.compile(r"time[d ]?out|timed out", re.I),
    "HTTP_5XX": re.compile(
        r"\bHTTP[ /]?5\d\d\b|\b5\d\d (?:Service Unavailable|Internal Server Error|Bad Gateway|Gateway Timeout)|returning http 5\d\d",
        re.I,
    ),
    "DB": re.compile(r"\b(?:sql\w*|jdbc|database|db|oracle|postgres\w*|mysql|mongo\w*|query|queries|hikari\w*)\b|db-", re.I),
    "POOL": re.compile(r"pool|hikari|connection acquisition|idle object|active connections|connections:|obtain jdbc connection|waiting for connection", re.I),
    "CIRCUIT_BREAKER": re.compile(r"circuit[ -]?breaker|breaker (?:open|half-open)", re.I),
    "RETRY": re.compile(r"\bretr(?:y|ies|ying)\b", re.I),
    "RETRY_EXHAUSTED": re.compile(r"retr(?:y|ies) exhausted|max(?:imum)? retries|giving up", re.I),
    "DEPENDENCY_UNAVAILABLE": re.compile(
        r"(?:downstream|dependency|upstream)\b.{0,40}\b(?:unavailable|down|failed|timeout)|\bunavailable\b|unreachable|connection refused",
        re.I,
    ),
    "LEAK": re.compile(r"\bleak", re.I),
    "STARVATION": re.compile(r"starvation|exhaust(?:ed|ion)|exceeded threshold|saturat", re.I),
    "LATENCY": re.compile(r"latency|\bslow\b|delayed|response time|wait time|acquisition time", re.I),
    "ERROR_RATE": re.compile(r"error rate", re.I),
    "DEPLOYMENT": re.compile(r"\bdeploy(?:ment|ed|ing)?\b|\brollout\b|\brelease\b|version=|\bupgrade", re.I),
    "CONFIG_CHANGE": re.compile(r"config(?:uration)? (?:change|update|reload)|feature flag", re.I),
    "INCIDENT_DECLARED": re.compile(r"incident (?:declared|triggered|opened)|\bP[0-4]\b.*incident|major .*outage|page[sd]? on-?call", re.I),
    "REMEDIATION": re.compile(
        r"\brestart(?:ing|ed)?\b|rollback|roll back|failover|scal(?:ed|ing) (?:up|out)|auto-remediation|releasing|investigating",
        re.I,
    ),
    "RECOVERY": re.compile(
        r"resumed|restored|recovered|recovery|normali[sz]ed|\bhealthy\b|back to|succeeding|\bstable\b|reduced to|dropping",
        re.I,
    ),
    "INCIDENT_RESOLVED": re.compile(r"incident resolved|\bresolved\b|all clear", re.I),
    "DIAGNOSIS": re.compile(r"root cause|\bidentified\b|\bconfirmed\b|caused by|due to", re.I),
}


class EventProcessor:
    def __init__(self, burst_gap: timedelta = BURST_GAP):
        self.burst_gap = burst_gap

    # ------------------------------------------------------------------ #
    @staticmethod
    def template_of(event: LogEvent) -> Tuple[str, str]:
        masked = event.message
        for pattern, repl in _MASKS:
            masked = pattern.sub(repl, masked)
        masked = re.sub(r"\s+", " ", masked).strip()
        key = f"{event.service or ''}|{event.level or ''}|{event.exception_type or ''}|{masked}"
        return hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:10], masked

    @staticmethod
    def tags_of(event: LogEvent) -> frozenset:
        text = event.raw if event.continuation_lines else event.message
        tags = {name for name, pattern in TAG_PATTERNS.items() if pattern.search(text)}
        if event.exception_type or event.has_stack_trace:
            tags.add("EXCEPTION")
        if event.status_code and 500 <= event.status_code < 600:
            tags.add("HTTP_5XX")
        # Recovery wording on an ERROR line is not recovery ("recovery failed").
        if event.level in ("ERROR", "CRITICAL"):
            tags.discard("RECOVERY")
            tags.discard("INCIDENT_RESOLVED")
        return frozenset(tags)

    # ------------------------------------------------------------------ #
    def process(self, events: List[LogEvent], group_prefix: str = "") -> List[EventGroup]:
        """Assigns template/group/tags to every event (in place) and returns groups.

        `group_prefix` (e.g. a short document id) keeps groups of different
        uploaded files apart within one investigation.
        """
        groups: Dict[str, EventGroup] = {}
        for event in events:
            template_id, template = self.template_of(event)
            event.template_id = template_id
            event.group_id = f"G{group_prefix}{template_id}"
            event.tags = self.tags_of(event)
            group = groups.get(event.group_id)
            if group is None:
                group = EventGroup(
                    id=event.group_id,
                    template=template,
                    service=event.service,
                    level=event.level,
                    exception_type=event.exception_type,
                )
                groups[event.group_id] = group
            self._add(group, event)
        return list(groups.values())

    def _add(self, group: EventGroup, event: LogEvent) -> None:
        group.event_ids.append(event.id)
        group.seqs.append(event.seq)
        group.tags |= event.tags
        group.has_stack_trace = group.has_stack_trace or event.has_stack_trace
        if group.first_ts is None:
            group.first_ts = event.ts
        previous_ts = group.last_ts
        group.last_ts = event.ts or group.last_ts

        # Bursts: a new burst starts after a gap, so separate occurrences stay visible.
        if not group.bursts or (
            event.ts and previous_ts and event.ts - previous_ts > self.burst_gap
        ):
            group.bursts.append(Burst(event.id, event.id, 1, event.ts, event.ts))
        else:
            burst = group.bursts[-1]
            burst.end_event_id, burst.count, burst.end_ts = event.id, burst.count + 1, event.ts

        if event.ids:
            for key in ("request_id", "trace_id", "correlation_id", "txn_id"):
                value = event.ids.get(key)
                if value and value not in group.distinct_request_ids and len(group.distinct_request_ids) < MAX_DISTINCT_REQUEST_IDS:
                    group.distinct_request_ids[value] = event.id
                    break

        if event.metrics:
            for name, value in event.metrics.items():
                stats = group.metric_stats.get(name)
                if stats is None:
                    group.metric_stats[name] = {"first": value, "min": value, "max": value, "last": value}
                    if group.peak_event_id is None:
                        group.peak_event_id = event.id
                else:
                    if value > stats["max"]:
                        stats["max"] = value
                        if name == next(iter(group.metric_stats)):
                            group.peak_event_id = event.id
                    stats["min"] = min(stats["min"], value)
                    stats["last"] = value

    @staticmethod
    def stats(events: Iterable[LogEvent], groups: List[EventGroup]) -> Dict[str, object]:
        events = list(events)
        level_counts: Dict[str, int] = {}
        for e in events:
            level_counts[e.level or "UNKNOWN"] = level_counts.get(e.level or "UNKNOWN", 0) + 1
        timed = [e.ts for e in events if e.ts]
        return {
            "events": len(events),
            "multiline_events": sum(1 for e in events if e.is_multiline),
            "stack_trace_events": sum(1 for e in events if e.has_stack_trace),
            "templates": len(groups),
            "groups": len(groups),
            "level_counts": level_counts,
            "first_ts": min(timed).isoformat(sep=" ") if timed else None,
            "last_ts": max(timed).isoformat(sep=" ") if timed else None,
        }


def first_abnormal(events: Iterable[LogEvent], levels: Optional[frozenset] = None) -> Optional[LogEvent]:
    wanted = levels or frozenset({"WARN", "ERROR", "CRITICAL"})
    for event in events:
        if event.level in wanted:
            return event
    return None
