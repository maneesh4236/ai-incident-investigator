"""
Deterministic, explainable incident analysis (no LLM).

Used when Gemini is unavailable (RCA fallback, chat fallback) so the product
still produces a genuinely useful, evidence-grounded explanation.

Two parts:

1. Milestones - concepts that are often confused and are kept separate:
   earliest anomaly, earliest database symptom, first error, first service
   failure, first report of unavailability, first downstream propagation,
   first recovery signal, resolution.

2. Root-cause hypothesis - candidate mechanisms are taken ONLY from what the
   logs state (generic SRE vocabulary, nothing incident-specific):
     * leaks / deadlock / out-of-memory / disk full / expired certificate /
       misconfiguration named in a log line, and
     * deployments / config changes shortly before the first anomaly.
   Each candidate is scored with explicit, explainable evidence factors:
     explicit diagnostic statement (+0.30), independent corroboration (+0.10),
     a later "root cause confirmed/identified" line (+0.05), consistent
     resource pressure/exhaustion (+0.10), failure of the implicated service
     after the pressure (+0.05), propagation along stated dependencies
     (+0.05), recovery after remediation aimed at it (+0.10), explicit causal
     wording (+0.15, the only route to CONFIRMED); change candidates score
     precedence (+0.10) and rollback-then-recovery (+0.15).
   Confidence is capped at 0.75 without an explicit causal statement
   (LIKELY at most), and reduced when a competing candidate is close.

Time order is never treated as causation: the earliest anomaly is reported as
a precursor unless it is itself part of the strongest mechanism's evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.models.schemas import Claim, ClaimType, GraphRelationship, RelationType
from app.services.ingestion.events import ERROR_LEVELS, EventGroup, LogEvent

_ABNORMAL = ("WARN", "ERROR", "CRITICAL")
ALERTING_SERVICE_RE = re.compile(r"alert|monitor|pager|oncall|on-call|sre-|ops-bot|sre\b", re.I)
_DIAGNOSTIC_VERB_RE = re.compile(r"\b(detected|identified|confirmed|found|diagnosed|root cause)\b", re.I)
_ROOT_CAUSE_STATEMENT_RE = re.compile(r"root cause\b.*\b(confirmed|identified|found)|\b(confirmed|identified|found)\b.*root cause", re.I)
_CAUSAL_WORDING_RE = re.compile(r"\b(caused by|due to|because of|resulted in|led to)\b", re.I)
_FAILURE_WORDING_RE = re.compile(r"fail|unable|unavailable|exhaust|timeout|timed out|refused|error|exception|starvation", re.I)

_LEAK_RE = re.compile(
    r"\b(session|connection|memory|thread|socket|file[- ]descriptor|fd|handle|cursor)s?\s+leak", re.I
)
_RESOURCE_TERMS: Dict[str, Set[str]] = {
    "session": {"session", "connection", "pool", "jdbc", "hikari"},
    "connection": {"connection", "session", "pool", "jdbc", "hikari"},
    "cursor": {"cursor", "session", "connection"},
    "memory": {"memory", "heap", "gc", "oom"},
    "thread": {"thread", "executor", "pool"},
    "socket": {"socket", "connection", "port"},
    "file-descriptor": {"file", "descriptor", "fd", "socket"},
    "fd": {"file", "descriptor", "fd", "socket"},
    "handle": {"handle"},
}
_OTHER_MECHANISMS: List[Tuple[re.Pattern, str, Set[str]]] = [
    (re.compile(r"\bdeadlock", re.I), "deadlock", {"lock", "deadlock", "transaction"}),
    (re.compile(r"out of memory|outofmemoryerror|\boom\b", re.I), "out-of-memory condition", {"memory", "heap", "gc", "oom"}),
    (re.compile(r"disk (?:is )?full|no space left", re.I), "full disk", {"disk", "space", "write"}),
    (re.compile(r"certificate (?:has )?expired|cert(?:ificate)? (?:invalid|expired)", re.I), "expired certificate",
     {"tls", "ssl", "certificate", "handshake"}),
    (re.compile(r"misconfigur|invalid configuration|config(?:uration)? error", re.I), "misconfiguration", {"config"}),
]
_ROLLBACK_RE = re.compile(r"rollback|roll back|revert", re.I)


# --------------------------------------------------------------------------- #
@dataclass
class Factor:
    label: str
    weight: float
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    mechanism: str
    service: Optional[str]
    kind: str  # "mechanism" | "change"
    finding_ids: List[str]
    resource_terms: Set[str]
    factors: List[Factor] = field(default_factory=list)
    pressure_ids: List[str] = field(default_factory=list)
    exhaustion_ids: List[str] = field(default_factory=list)
    failure_id: Optional[str] = None
    propagation: List[Tuple[str, str]] = field(default_factory=list)  # (service, event id)
    remediation_ids: List[str] = field(default_factory=list)
    recovery_ids: List[str] = field(default_factory=list)
    explicit_causal_ids: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    claim_type: ClaimType = ClaimType.UNKNOWN

    @property
    def score(self) -> float:
        return round(sum(f.weight for f in self.factors), 2)

    def evidence_ids(self) -> List[str]:
        ids: List[str] = []
        for group in (self.finding_ids, self.pressure_ids, self.exhaustion_ids, [self.failure_id] if self.failure_id else [],
                      [eid for _, eid in self.propagation], self.remediation_ids, self.recovery_ids):
            for eid in group:
                if eid and eid not in ids:
                    ids.append(eid)
        return ids


@dataclass
class IncidentAnalysis:
    earliest_anomaly: Optional[LogEvent] = None
    earliest_db_symptom: Optional[LogEvent] = None
    first_error: Optional[LogEvent] = None
    first_service_failure: Optional[LogEvent] = None
    first_unavailability_report: Optional[LogEvent] = None
    first_propagation: Optional[LogEvent] = None
    first_recovery: Optional[LogEvent] = None
    resolution: Optional[LogEvent] = None
    hypothesis: Optional[Hypothesis] = None
    alternatives: List[Hypothesis] = field(default_factory=list)
    lookup: Callable[[str], Optional[LogEvent]] = lambda _eid: None

    # ------------------------------------------------------------------ #
    @property
    def confidence_label(self) -> str:
        return confidence_label(self.hypothesis.confidence if self.hypothesis else 0.15)

    def precursor(self) -> Optional[LogEvent]:
        """Earliest anomaly, when it is NOT part of the strongest mechanism's evidence."""
        anomaly = self.earliest_anomaly
        if anomaly is None or self.hypothesis is None:
            return anomaly
        return None if anomaly.id in self.hypothesis.evidence_ids() else anomaly

    def root_cause_statement(self) -> str:
        h = self.hypothesis
        if h is None:
            anomaly, failure = self.earliest_anomaly, self.first_service_failure
            if anomaly is None:
                return ("No WARN/ERROR events were found. Evidence is insufficient to establish an incident or its "
                        "root cause.")
            text = (f"Evidence is insufficient to establish a root cause: no log line names a causal mechanism. "
                    f"Earliest anomaly: {cite(anomaly)}.")
            if failure is not None and failure.id != anomaly.id:
                text += f" First service failure: {cite(failure)}."
            return text + " The order of these events is a temporal sequence, not established causation."

        where = f" in {h.service}" if h.service else ""
        type_text = f"{h.claim_type.value}, {confidence_label(h.confidence)} confidence"
        lead = "Root cause" if h.claim_type == ClaimType.CONFIRMED else "Likely root cause" if h.claim_type == ClaimType.LIKELY else "Possible root cause"
        parts = [f"{lead} ({type_text}): {h.mechanism}{where} [{', '.join(h.finding_ids[:3])}]."]
        if h.kind == "change":
            if h.recovery_ids:
                parts.append(f"Recovery followed the rollback/revert [{', '.join((h.remediation_ids + h.recovery_ids)[:3])}].")
        else:
            mechanism_detail = []
            if h.pressure_ids:
                mechanism_detail.append(f"{_resource_phrase(h).lower()} pressure building{where} [{', '.join(h.pressure_ids[:2])}]")
            if h.exhaustion_ids:
                mechanism_detail.append(f"reaching exhaustion/timeouts [{', '.join(h.exhaustion_ids[:4])}]")
            if mechanism_detail:
                parts.append("The logs show " + " and ".join(mechanism_detail) + ".")
            if h.failure_id:
                failure = self.lookup(h.failure_id)
                parts.append(f"{(h.service or 'The service')} then failed ({cite(failure)}).")
            if h.propagation:
                services = ", ".join(f"{svc} [{eid}]" for svc, eid in h.propagation[:4])
                parts.append(f"Failures propagated to dependent services: {services}.")
            if h.remediation_ids and h.recovery_ids:
                parts.append(f"Recovery followed remediation aimed at {h.service or 'the affected resource'} "
                             f"[{', '.join(h.remediation_ids[:2])}] (recovery signals [{', '.join(h.recovery_ids[:2])}]).")
        precursor = self.precursor()
        if precursor is not None:
            parts.append(f"The earlier \"{precursor.message}\" [{precursor.id}] appears to be an early anomaly/precursor "
                         "rather than the strongest root-cause evidence; the logs do not establish whether it was a "
                         "cause or an effect.")
        if h.claim_type != ClaimType.CONFIRMED:
            parts.append("No log line explicitly states the causal link, so this is not CONFIRMED.")
        return " ".join(parts)

    def confidence_explanation(self) -> str:
        h = self.hypothesis
        if h is None:
            return ("Low: no log line names a causal mechanism; only the temporal order of anomalies and failures "
                    "is observed.")
        reasons = [f.label for f in h.factors if f.weight > 0]
        penalties = [f.label for f in h.factors if f.weight < 0]
        text = f"{confidence_label(h.confidence)} ({h.confidence:.2f}) because " + "; ".join(reasons) if reasons else ""
        if penalties:
            text += ". Reduced because " + "; ".join(penalties)
        if h.limitations:
            text += ". Limitations: " + "; ".join(h.limitations)
        return text + "."

    def chain(self) -> List[dict]:
        """Cause/failure chain with claim types (OBSERVED steps, LIKELY link)."""
        h = self.hypothesis
        steps: List[dict] = []
        precursor = self.precursor()
        if precursor is not None:
            steps.append({"step": f"Early anomaly (precursor): {precursor.message}", "type": ClaimType.OBSERVED.value,
                          "evidence_ids": [precursor.id]})
        if h is None:
            if self.first_service_failure is not None:
                steps.append({"step": f"First service failure: {self.first_service_failure.service}: "
                                      f"{self.first_service_failure.message}", "type": ClaimType.OBSERVED.value,
                              "evidence_ids": [self.first_service_failure.id]})
            if self.first_propagation is not None:
                steps.append({"step": f"Propagation: {self.first_propagation.service}: {self.first_propagation.message}",
                              "type": ClaimType.OBSERVED.value, "evidence_ids": [self.first_propagation.id]})
            return steps
        steps.append({"step": f"{h.mechanism[0].upper()}{h.mechanism[1:]} reported"
                              + (f" in {h.service}" if h.service else ""),
                      "type": ClaimType.OBSERVED.value, "evidence_ids": h.finding_ids[:3]})
        if h.pressure_ids or h.exhaustion_ids:
            steps.append({"step": f"{_resource_phrase(h)} pressure and exhaustion", "type": ClaimType.OBSERVED.value,
                          "evidence_ids": (h.pressure_ids[:1] + h.exhaustion_ids[:3])})
        if h.failure_id:
            failure = self.lookup(h.failure_id)
            steps.append({"step": f"{h.service or 'Service'} failure: {failure.message if failure else ''}",
                          "type": ClaimType.OBSERVED.value, "evidence_ids": [h.failure_id]})
        if h.propagation:
            steps.append({"step": "Propagation to " + ", ".join(s for s, _ in h.propagation[:4]),
                          "type": ClaimType.OBSERVED.value, "evidence_ids": [eid for _, eid in h.propagation[:4]]})
        if h.recovery_ids:
            steps.append({"step": "Recovery after remediation", "type": ClaimType.OBSERVED.value,
                          "evidence_ids": (h.remediation_ids[:1] + h.recovery_ids[:2])})
        return steps

    def claims(self) -> List[Claim]:
        out: List[Claim] = []

        def add(text: str, ctype: ClaimType, ids: Iterable[str]) -> None:
            ids = [i for i in ids if i and self.lookup(i) is not None]
            if ctype in (ClaimType.OBSERVED, ClaimType.CONFIRMED) and not ids:
                ctype = ClaimType.UNKNOWN
            out.append(Claim(text=text, type=ctype, evidence_ids=ids, citations_valid=bool(ids)))

        for label, event in (("Earliest anomaly", self.earliest_anomaly), ("First service failure", self.first_service_failure),
                             ("First downstream propagation", self.first_propagation), ("First recovery signal", self.first_recovery)):
            if event is not None:
                add(f"{label}: {event.service or ''} {event.message}".strip(), ClaimType.OBSERVED, [event.id])
        h = self.hypothesis
        if h is not None:
            add(f"The logs state: {h.mechanism}" + (f" in {h.service}" if h.service else ""), ClaimType.OBSERVED, h.finding_ids[:3])
            if h.exhaustion_ids:
                add(f"{_resource_phrase(h)} exhaustion and timeouts followed", ClaimType.INFERRED, h.exhaustion_ids[:3])
            add(f"{h.mechanism} is the most likely underlying cause of the outage", h.claim_type, h.finding_ids[:2] + h.exhaustion_ids[:1])
        else:
            add("Root cause", ClaimType.UNKNOWN, [])
        return out

    def milestones_dict(self) -> Dict[str, Optional[dict]]:
        def describe(event: Optional[LogEvent]) -> Optional[dict]:
            if event is None:
                return None
            return {"event_id": event.id, "timestamp": event.ts_raw, "service": event.service,
                    "level": event.level, "message": event.message}

        return {
            "earliest_anomaly": describe(self.earliest_anomaly),
            "earliest_db_symptom": describe(self.earliest_db_symptom),
            "first_error": describe(self.first_error),
            "first_service_failure": describe(self.first_service_failure),
            "first_unavailability_report": describe(self.first_unavailability_report),
            "first_propagation": describe(self.first_propagation),
            "first_recovery": describe(self.first_recovery),
            "resolution": describe(self.resolution),
            "precursor": describe(self.precursor()),
        }


# --------------------------------------------------------------------------- #
def confidence_label(confidence: float) -> str:
    if confidence >= 0.65:
        return "High"
    if confidence >= 0.4:
        return "Medium"
    return "Low"


def cite(event: Optional[LogEvent]) -> str:
    if event is None:
        return "(event unavailable)"
    service = f"{event.service} " if event.service else ""
    return f"[{event.id}] {event.ts_display} {service}{event.level or ''}: {event.message}"


def _resource_phrase(h: Hypothesis) -> str:
    key = h.mechanism.split()[0].lower() if h.mechanism else "resource"
    if key in ("session", "connection"):
        return "Connection/session"
    return {"memory": "Memory", "thread": "Thread", "socket": "Socket", "cursor": "Cursor"}.get(key, "Resource")


def _ts_key(event: Optional[LogEvent]) -> tuple:
    if event is None:
        return (1, datetime.max, 0)
    return (0, event.ts or datetime.max, event.seq)


def _mentions(text: str, terms: Iterable[str]) -> bool:
    low = text.lower()
    for term in terms:
        if term in ("hikari", "pool", "jdbc"):  # often embedded in CamelCase names (DbPool, HikariPool)
            if term in low:
                return True
        elif re.search(rf"\b{re.escape(term)}s?\b", low):
            return True
    return False


# --------------------------------------------------------------------------- #
def analyze_incident(
    groups: Sequence[EventGroup],
    lookup: Callable[[str], Optional[LogEvent]],
    graph_facts: Sequence[GraphRelationship] = (),
) -> IncidentAnalysis:
    """Builds milestones and the strongest evidence-backed root-cause hypothesis."""
    firsts: List[Tuple[EventGroup, LogEvent]] = []
    for g in groups:
        event = lookup(g.first_event_id) if g.event_ids else None
        if event is not None:
            firsts.append((g, event))
    firsts.sort(key=lambda ge: _ts_key(ge[1]))
    analysis = IncidentAnalysis(lookup=lookup)
    if not firsts:
        return analysis

    def is_alerting(event: LogEvent) -> bool:
        return bool(event.service and ALERTING_SERVICE_RE.search(event.service))

    abnormal = [(g, e) for g, e in firsts if e.level in _ABNORMAL and not is_alerting(e)]
    services = {e.service for _, e in firsts if e.service}

    analysis.earliest_anomaly = abnormal[0][1] if abnormal else None
    analysis.earliest_db_symptom = next((e for g, e in abnormal if "DB" in g.tags), None)
    analysis.first_error = next((e for g, e in firsts if e.level in ERROR_LEVELS), None)
    failures = [(g, e) for g, e in abnormal if e.level in ERROR_LEVELS and not (g.tags & {"INCIDENT_DECLARED", "DIAGNOSIS"})]
    analysis.first_service_failure = failures[0][1] if failures else None
    failing_service = analysis.first_service_failure.service if analysis.first_service_failure else None
    analysis.first_unavailability_report = next(
        (e for g, e in abnormal if "DEPENDENCY_UNAVAILABLE" in g.tags or re.search(r"unavailable", e.message, re.I)), None)
    if failing_service:
        stem = failing_service.split("-")[0].lower()
        analysis.first_propagation = next(
            (e for g, e in failures if e.service and e.service != failing_service
             and (g.tags & {"DEPENDENCY_UNAVAILABLE", "HTTP_5XX"} or stem in e.message.lower())), None)
    after_failure = (lambda e: analysis.first_service_failure is None
                     or _ts_key(e) >= _ts_key(analysis.first_service_failure))
    recovery = [(g, e) for g, e in firsts if g.tags & {"RECOVERY", "INCIDENT_RESOLVED"} and e.level not in ERROR_LEVELS
                and after_failure(e)]
    analysis.first_recovery = recovery[0][1] if recovery else None
    analysis.resolution = next((e for g, e in recovery if "INCIDENT_RESOLVED" in g.tags), None)

    candidates = _candidates(firsts, services, analysis)
    for h in candidates:
        _score(h, firsts, abnormal, services, analysis, graph_facts, lookup)
    candidates = [h for h in candidates if h.score > 0]
    candidates.sort(key=lambda h: -h.score)
    if not candidates:
        return analysis

    best = candidates[0]
    if len(candidates) > 1 and candidates[1].score >= best.score - 0.1:
        best.factors.append(Factor(f"a competing explanation ({candidates[1].mechanism}) has similar support", -0.1))
    cap = 0.9 if best.explicit_causal_ids else 0.75
    best.confidence = round(max(0.0, min(best.score, cap)), 2)
    if best.explicit_causal_ids:
        best.claim_type = ClaimType.CONFIRMED
    elif best.confidence >= 0.5:
        best.claim_type = ClaimType.LIKELY
    elif best.confidence >= 0.25:
        best.claim_type = ClaimType.INFERRED
    else:
        return analysis
    analysis.hypothesis = best
    analysis.alternatives = candidates[1:3]
    return analysis


def _candidates(firsts, services: Set[str], analysis: IncidentAnalysis) -> List[Hypothesis]:
    by_key: Dict[str, Hypothesis] = {}
    for g, e in firsts:
        text = e.message
        found: Optional[Tuple[str, Set[str]]] = None
        leak = _LEAK_RE.search(text)
        if leak:
            subject = leak.group(1).lower().replace(" ", "-")
            found = (f"{leak.group(1).lower()} leak", _RESOURCE_TERMS.get(subject, {subject}))
        else:
            for pattern, label, terms in _OTHER_MECHANISMS:
                if pattern.search(text):
                    found = (label, terms)
                    break
        if found is None:
            continue
        key, terms = found
        h = by_key.get(key)
        if h is None:
            h = by_key[key] = Hypothesis(mechanism=key, service=None, kind="mechanism", finding_ids=[], resource_terms=set(terms))
        h.finding_ids.extend(eid for eid in g.instance_ids(3) if eid not in h.finding_ids)
        mentioned = next((s for s in services if s and re.search(rf"\b{re.escape(s)}\b", text)), None)
        if h.service is None and mentioned:
            h.service = mentioned

    # Changes shortly before the first anomaly (deployment / config change).
    anomaly = analysis.earliest_anomaly
    if anomaly is not None:
        for g, e in firsts:
            if g.tags & {"DEPLOYMENT", "CONFIG_CHANGE"} and _ts_key(e) <= _ts_key(anomaly):
                if anomaly.ts and e.ts and (anomaly.ts - e.ts).total_seconds() > 3600:
                    continue
                key = "deployment" if "DEPLOYMENT" in g.tags else "configuration change"
                if key not in by_key:
                    by_key[key] = Hypothesis(mechanism=f"{key} ({e.message})", service=e.service, kind="change",
                                             finding_ids=[e.id], resource_terms=set())
    return list(by_key.values())


def _score(h: Hypothesis, firsts, abnormal, services, analysis: IncidentAnalysis, graph_facts, lookup) -> None:
    findings = [lookup(eid) for eid in h.finding_ids]
    findings = [f for f in findings if f is not None]
    first_failure = analysis.first_service_failure

    if h.kind == "change":
        change = findings[0] if findings else None
        if change is not None:
            h.factors.append(Factor("a change was deployed shortly before the first anomaly", 0.10, [change.id]))
        remediation = [e for g, e in firsts if "REMEDIATION" in g.tags and _ROLLBACK_RE.search(e.message)]
        recovery = [e for g, e in firsts if g.tags & {"RECOVERY", "INCIDENT_RESOLVED"} and e.level not in ERROR_LEVELS
                    and remediation and _ts_key(e) > _ts_key(remediation[0])]
        if remediation and recovery:
            h.remediation_ids = [remediation[0].id]
            h.recovery_ids = [r.id for r in recovery[:2]]
            h.factors.append(Factor("the system recovered after the change was rolled back", 0.15, h.remediation_ids + h.recovery_ids))
        h.limitations.append("no log line states that the change caused the failure")
        return

    sources = {f.service for f in findings if f.service}
    if any(_DIAGNOSTIC_VERB_RE.search(f.message) for f in findings):
        h.factors.append(Factor(f"the logs explicitly report the {h.mechanism}", 0.30, h.finding_ids[:2]))
    else:
        h.factors.append(Factor(f"the logs mention a {h.mechanism}", 0.20, h.finding_ids[:2]))
    if len(sources) >= 2:
        h.factors.append(Factor(f"it is corroborated by {len(sources)} independent sources ({', '.join(sorted(sources))})",
                                0.10, h.finding_ids[:3]))
    first_finding = min(findings, key=_ts_key) if findings else None
    statements = [e for g, e in firsts if _ROOT_CAUSE_STATEMENT_RE.search(e.message)
                  and (first_finding is None or _ts_key(e) >= _ts_key(first_finding))]
    if statements:
        h.factors.append(Factor("a later log line states that a root cause was confirmed/identified", 0.05,
                                [statements[0].id]))

    related_sources = {h.service} | sources
    related_sources.discard(None)

    def related(e: LogEvent) -> bool:
        return (e.service in related_sources and not ALERTING_SERVICE_RE.search(e.service or "")) or (
            h.service is not None and h.service in e.message)

    resource_events = [(g, e) for g, e in abnormal if related(e) and _mentions(e.message, h.resource_terms)
                       and e.id not in h.finding_ids]
    h.pressure_ids = [e.id for g, e in resource_events if e.level == "WARN"][:3]
    h.exhaustion_ids = [e.id for g, e in resource_events if e.level in ERROR_LEVELS
                        and (g.tags & {"STARVATION", "TIMEOUT", "POOL"} or _FAILURE_WORDING_RE.search(e.message))][:5]
    if h.pressure_ids or h.exhaustion_ids:
        h.factors.append(Factor(f"{_resource_phrase(h).lower()} pressure and exhaustion are consistent with it",
                                0.10, (h.pressure_ids[:1] + h.exhaustion_ids[:3])))

    service_failures = [e for g, e in abnormal if e.level in ERROR_LEVELS and h.service and e.service == h.service
                        and not (g.tags & {"INCIDENT_DECLARED", "DIAGNOSIS"})]
    if service_failures:
        h.failure_id = service_failures[0].id
        h.exhaustion_ids = [eid for eid in h.exhaustion_ids if eid != h.failure_id]
        pressure_first = lookup(h.pressure_ids[0]) if h.pressure_ids else None
        if pressure_first is None or _ts_key(pressure_first) <= _ts_key(service_failures[0]):
            h.factors.append(Factor(f"{h.service} failed after the pressure built up", 0.05, [h.failure_id]))
        if first_finding is not None and _ts_key(first_finding) > _ts_key(service_failures[0]):
            h.limitations.append(f"the {h.mechanism} was only detected after failures began, so the logs do not show "
                                 "when it started")

    if h.service:
        stem = h.service.split("-")[0].lower()
        deps = {r.source: (r.evidence_event_ids or [None])[0] for r in graph_facts
                if r.type == RelationType.DEPENDS_ON and r.target == h.service and r.basis == "explicit_text"}
        prop: List[Tuple[str, str]] = []
        for g, e in abnormal:
            if e.level in ERROR_LEVELS and e.service and e.service != h.service and not ALERTING_SERVICE_RE.search(e.service) \
                    and e.service not in sources and (e.service in deps or "DEPENDENCY_UNAVAILABLE" in g.tags
                                                      or "HTTP_5XX" in g.tags or stem in e.message.lower()):
                if e.service not in [s for s, _ in prop]:
                    prop.append((e.service, e.id))
        h.propagation = prop
        if prop:
            h.factors.append(Factor(f"failures then propagated to {len(prop)} dependent service(s)", 0.05,
                                    [eid for _, eid in prop[:3]]))

    remediation = [e for g, e in firsts if "REMEDIATION" in g.tags and e.level not in ERROR_LEVELS
                   and ((h.service and h.service in e.message) or _mentions(e.message, h.resource_terms)
                        or (h.service and e.service == h.service))]
    if remediation:
        recovery = [e for g, e in firsts if g.tags & {"RECOVERY", "INCIDENT_RESOLVED"} and e.level not in ERROR_LEVELS
                    and _ts_key(e) > _ts_key(remediation[0])]
        resource_recovery = [e for e in recovery if _mentions(e.message, h.resource_terms)]
        if recovery:
            h.remediation_ids = [r.id for r in remediation[:3]]
            h.recovery_ids = [r.id for r in (resource_recovery + [x for x in recovery if x not in resource_recovery])[:3]]
            h.factors.append(Factor("the system recovered after remediation aimed at it", 0.10,
                                    h.remediation_ids[:2] + h.recovery_ids[:2]))

    for f in findings:
        if _CAUSAL_WORDING_RE.search(f.message):
            h.explicit_causal_ids.append(f.id)
    for r in graph_facts:
        if r.is_causal and h.mechanism.split()[0] in r.source.lower():
            h.explicit_causal_ids.extend(i for i in r.evidence_event_ids if lookup(i) is not None)
    if h.explicit_causal_ids:
        h.factors.append(Factor("a log line explicitly states the causal link", 0.15, h.explicit_causal_ids[:2]))
