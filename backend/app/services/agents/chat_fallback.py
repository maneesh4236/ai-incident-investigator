"""
Deterministic, question-aware chat answers used ONLY when the single Gemini
chat call fails (503 / timeout / transport / invalid output / not configured).

No LLM is involved: the answer is assembled from the same evidence the Gemini
call would have received (the chat EvidencePack), the investigation's event
groups, the timeline phase rules and evidence-backed graph facts.

Safety rules enforced here:
  * every cited id is a real event id from the event store (or a D##### id that
    was in the evidence pack); nothing is invented - services, timestamps and
    messages are copied from parsed events;
  * time order is reported as OBSERVED sequence, never as causation;
  * causal language is used only for evidence-backed causal edges
    (CAUSES/TRIGGERS with basis explicit_text/llm_cited) or log lines that
    themselves state a diagnosis - RELATED_TO / PRECEDES are never causal;
  * the answer states explicitly when causation is not established.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence

from app.models.schemas import Claim, ClaimType, GraphRelationship, RCAReport, RelationType, TimelinePhase
from app.services.ingestion.events import ERROR_LEVELS, EventGroup, LogEvent
from app.services.reasoning.evidence_selector import EvidencePack
from app.services.reasoning.incident_analysis import analyze_incident
from app.services.reasoning.timeline_builder import TimelineBuilder

_ALERTING_SERVICE_RE = re.compile(r"alert|monitor|pager|oncall|on-call|sre-", re.I)
_IMPACT_STATEMENT_RE = re.compile(r"impact|error rate|outage|customers?|users?|availability", re.I)
_CONTEXT_TAGS = {"INCIDENT_DECLARED", "REMEDIATION", "DIAGNOSIS"}
_RECOVERY_TAGS = {"RECOVERY", "INCIDENT_RESOLVED"}
_PHASE_ORDER = [
    TimelinePhase.PRECURSOR, TimelinePhase.ANOMALY, TimelinePhase.DEGRADATION,
    TimelinePhase.FAILURE, TimelinePhase.PROPAGATION, TimelinePhase.RECOVERY,
]

# Ordered: the first matching intent wins.
_INTENTS = [
    ("related", re.compile(r"(related|similar|past|previous|other|prior)\s+(incidents?|outages?)|seen this before", re.I)),
    ("chain", re.compile(r"cause[- ]?chain|causal chain|failure chain|chain of (events|causes)|\bchain\b", re.I)),
    ("start", re.compile(r"\b(first|earliest|start(ed|s)?|begin|began|onset|initial(ly)?)\b", re.I)),
    ("recovery", re.compile(r"recover|resolv|restor|mitigat|back to normal|\bfix(ed)?\b|remediat", re.I)),
    ("impact", re.compile(r"affect|impact|blast radius|customer|which services|what services|who was", re.I)),
    ("cause", re.compile(r"\bwhy\b|root[- ]cause|\bcause[sd]?\b|\breason\b|led to|trigger", re.I)),
    ("errors", re.compile(r"\berrors?\b|exceptions?|what failed|\bfail(ed|ure|ures|ing)?\b", re.I)),
    ("evidence", re.compile(r"evidence|proof|support|what do the logs", re.I)),
    ("timeline", re.compile(r"timeline|unfold|sequence|what happened|happen|walk me through|summar|overview|explain", re.I)),
]

_FRIENDLY_REASON = {
    "server_error": "Gemini is temporarily unavailable (server error / high demand)",
    "timeout": "Gemini did not respond in time",
    "network": "Gemini could not be reached (network error)",
    "rate_limited": "Gemini is rate-limited right now",
    "quota_exhausted": "the Gemini quota is exhausted",
    "unavailable": "Gemini is not accepting requests right now",
    "invalid_json": "Gemini returned an unusable response",
    "truncated": "Gemini's answer was cut off before it was complete",
    "invalid_llm_output": "Gemini returned an unusable response",
    "empty": "Gemini returned an empty response",
    "blocked": "Gemini declined to answer",
    "not_configured": "Gemini is not configured",
    "client_error": "the Gemini request was rejected",
    "question_too_long": "the question was too long to send with evidence",
}


def friendly_reason(kind: Optional[str]) -> str:
    return _FRIENDLY_REASON.get(kind or "", "AI reasoning is unavailable")


def classify_intent(question: str) -> str:
    for name, pattern in _INTENTS:
        if pattern.search(question or ""):
            return name
    return "generic"


@dataclass
class FallbackAnswer:
    answer: str
    intent: str
    claims: List[Claim] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)


class DeterministicChatAnswerer:
    def __init__(
        self,
        question: str,
        pack: EvidencePack,
        groups: Sequence[EventGroup],
        lookup: Callable[[str], Optional[LogEvent]],
        report: Optional[RCAReport] = None,
        graph_facts: Sequence[GraphRelationship] = (),
    ):
        self.question = question
        self.pack = pack
        self.groups = sorted((g for g in groups if g.event_ids), key=lambda g: _seq(lookup(g.first_event_id)))
        self.lookup = lookup
        self.report = report
        self.graph_facts = list(graph_facts)
        self.claims: List[Claim] = []
        self.cited: List[str] = []
        self.items = list(pack.items)
        self.phase: Dict[str, TimelinePhase] = self._phases()
        # Same deterministic analysis as the RCA fallback, over ALL event groups.
        self.analysis = analyze_incident(self.groups, lookup, self.graph_facts)

    # ------------------------------------------------------------------ #
    def answer(self, reason: Optional[str]) -> FallbackAnswer:
        intent = classify_intent(self.question)
        if not self.items and not self.pack.document_evidence:
            body = "No log evidence is available for this investigation, so this question cannot be answered."
        elif not self.items:
            body = self._documents_only()
        else:
            body = getattr(self, f"_{intent}")()
        footer = (
            f"\n\n(Note: {friendly_reason(reason)}, so this answer was generated deterministically from the "
            "investigation's log evidence. Event order shows sequence, not proven causation.)"
        )
        return FallbackAnswer(answer=body + footer, intent=intent, claims=self.claims, evidence_ids=self.cited)

    # ------------------------------------------------------------------ #
    # Intent handlers
    # ------------------------------------------------------------------ #
    def _cause(self) -> str:
        a = self.analysis
        lines = []
        report_line = self._report_root_cause()
        if report_line:
            lines.append(report_line)
        lines.append(a.root_cause_statement())
        self._cite_text(lines[-1])
        for claim in a.claims():
            self._claim(claim.text, claim.type, claim.evidence_ids)
        lines.append(f"Confidence: {a.confidence_explanation()}")
        causal = self._explicit_causal_facts()
        if causal:
            lines.append(causal)
        deps = self._dependency_facts()
        if deps:
            lines.append(deps)
        if a.hypothesis is None:
            lines.append(self._causation_caveat())
        return "\n".join(lines)

    def _chain(self) -> str:
        a = self.analysis
        steps = a.chain()
        if not steps:
            return "No abnormal events were found, so no cause chain can be built from the evidence."
        title = ("Failure chain built from the evidence (each step is OBSERVED in the logs):" if a.hypothesis
                 else "Observed incident chain (in time order):")
        lines = [title]
        for i, step in enumerate(steps, start=1):
            ids = [eid for eid in step["evidence_ids"] if self.lookup(eid) is not None]
            lines.append(f"  {i}. [{step['type']}] {step['step']} [{', '.join(ids)}]")
            self._claim(step["step"], ClaimType(step["type"]), ids)
        causal = self._explicit_causal_facts()
        if causal:
            lines.append(causal)
        if a.hypothesis is not None and a.hypothesis.claim_type != ClaimType.CONFIRMED:
            lines.append(f"The links between these steps are {a.hypothesis.claim_type.value} (corroborated, but no log line "
                         "explicitly states the causal link).")
        elif a.hypothesis is None:
            lines.append("No log line names a causal mechanism; the steps are time order (OBSERVED), not established "
                         "causation.")
        return "\n".join(lines)

    def _related(self) -> str:
        lines = ["Only this incident's logs are available to the investigator, so related past incidents cannot be "
                 "identified from the evidence."]
        h = self.analysis.hypothesis
        if h is not None:
            ids = h.evidence_ids()[:6]
            lines.append(f"Within this incident, the events most closely related to the {h.mechanism} are: "
                         + "; ".join(self._fmt(self.lookup(eid)) for eid in ids if self.lookup(eid)) + ".")
            self._claim(f"Events related to the {h.mechanism}", ClaimType.OBSERVED, ids)
        else:
            lines.append(self._generic())
        return "\n".join(lines)

    def _start(self) -> str:
        a = self.analysis
        anomaly, failure = a.earliest_anomaly, a.first_service_failure
        if anomaly is None:
            return "No WARN or ERROR events were found; the evidence does not show when an incident started."
        lines = []
        if failure is not None:
            lines.append(f"{failure.service or 'The first failing component'} was the first service to fail: "
                         f"{self._fmt(failure)}.")
            self._claim(f"First service failure: {failure.service}", ClaimType.OBSERVED, [failure.id])
        else:
            lines.append("No ERROR-level service failure was found in the evidence.")
        if anomaly.id != (failure.id if failure else None):
            kind = "a warning (anomaly), not a failure" if anomaly.level not in ERROR_LEVELS else "an error"
            lines.append(f"The earliest anomaly came before that: {self._fmt(anomaly)} - {kind}.")
            self._claim("Earliest anomaly", ClaimType.OBSERVED, [anomaly.id])
            gap = _gap(anomaly, failure) if failure is not None else ""
            if gap:
                lines.append(f"The first service failure followed the earliest anomaly by {gap}.")
        db = a.earliest_db_symptom
        if db is not None and db.id not in (anomaly.id, failure.id if failure else None):
            lines.append(f"Earliest database symptom: {self._fmt(db)}.")
            self._claim("Earliest database symptom", ClaimType.OBSERVED, [db.id])
        elif db is not None and db.id == anomaly.id:
            lines.append("That earliest anomaly is also the earliest database symptom.")
        first_error = a.first_error
        if first_error is not None and first_error.id not in ([failure.id] if failure else []):
            lines.append(f"First ERROR line of any kind: {self._fmt(first_error)}.")
        report = a.first_unavailability_report
        if report is not None and report.id not in (anomaly.id, failure.id if failure else None):
            lines.append(f"First report of unavailability: {self._fmt(report)}.")
            self._claim("First unavailability report", ClaimType.OBSERVED, [report.id])
        prop = a.first_propagation
        if prop is not None and (report is None or prop.id != report.id):
            lines.append(f"First downstream propagation: {self._fmt(prop)}.")
            self._claim("First downstream propagation", ClaimType.OBSERVED, [prop.id])
        elif prop is not None:
            lines.append("That is also the first downstream propagation (a dependent service failing).")
        lines.append("Being first in time does not by itself make an event the root cause.")
        return "\n".join(lines)

    def _recovery(self) -> str:
        first_failure = self._first_failure() or self._first_abnormal()
        after = (lambda e: first_failure is None or _seq(e) >= _seq(first_failure))
        actions = [self._first_event(g) for g in self.groups if "REMEDIATION" in g.tags and g.level not in ERROR_LEVELS]
        actions = [e for e in actions if e is not None and after(e)]
        signals = [self._first_event(g) for g in self.groups if g.tags & _RECOVERY_TAGS and g.level not in ERROR_LEVELS]
        signals = [e for e in signals if e is not None and after(e)]
        if not actions and not signals:
            return ("The evidence contains no recovery or resolution events, so it does not show whether or when "
                    "the system recovered.")
        lines = []
        if actions:
            lines.append("Remediation actions observed: " + "; ".join(self._fmt(e) for e in actions[:5]) + ".")
            self._claim("Remediation actions", ClaimType.OBSERVED, [e.id for e in actions[:5]])
        if signals:
            lines.append("Recovery signals observed: " + "; ".join(self._fmt(e) for e in signals[:8]) + ".")
            self._claim("Recovery signals", ClaimType.OBSERVED, [e.id for e in signals[:8]])
            first_rec, last_rec = signals[0], signals[-1]
            lines.append(f"Recovery began at {first_rec.ts_display} [{first_rec.id}] and the last recovery signal is at "
                         f"{last_rec.ts_display} [{last_rec.id}].")
            if first_failure is not None:
                span = _gap(first_failure, last_rec)
                if span:
                    lines.append(f"Time from the first failure [{first_failure.id}] to the last recovery signal: {span}.")
        last_abnormal = self._last_abnormal()
        if last_abnormal is not None:
            lines.append(f"The last abnormal event in the evidence is {self._fmt(last_abnormal)}.")
        if actions and signals:
            lines.append("Recovery followed the remediation steps in time; the logs do not by themselves prove the "
                         "remediation caused the recovery (INFERRED at most).")
        return "\n".join(lines)

    def _impact(self) -> str:
        failed: Dict[str, List[EventGroup]] = {}
        warned: Dict[str, List[EventGroup]] = {}
        for g in self.groups:
            if not g.service or _ALERTING_SERVICE_RE.search(g.service) or g.tags & _CONTEXT_TAGS:
                continue
            if g.level in ERROR_LEVELS:
                failed.setdefault(g.service, []).append(g)
            elif g.level == "WARN":
                warned.setdefault(g.service, []).append(g)
        lines = []
        if failed:
            lines.append("Services with ERROR events (failures observed):")
            for service, groups in failed.items():
                first = self._first_event(groups[0])
                total = sum(g.count for g in groups)
                lines.append(f"  - {service}: {total} error event(s) across {len(groups)} message type(s); first {self._fmt(first)}")
                self._claim(f"{service} logged errors", ClaimType.OBSERVED, [first.id])
        degraded_only = {s: gs for s, gs in warned.items() if s not in failed}
        if degraded_only:
            lines.append("Services with warnings only (degradation, no ERROR observed):")
            for service, groups in degraded_only.items():
                first = self._first_event(groups[0])
                lines.append(f"  - {service}: {sum(g.count for g in groups)} warning event(s); first {self._fmt(first)}")
                self._claim(f"{service} logged warnings", ClaimType.OBSERVED, [first.id])
        http5xx = [g for g in self.groups if "HTTP_5XX" in g.tags]
        if http5xx:
            parts = [f"{g.service or 'unknown'} x{g.count} (first [{g.first_event_id}])" for g in http5xx[:5]]
            lines.append("User-facing 5xx responses: " + "; ".join(parts) + ".")
            self._cite([g.first_event_id for g in http5xx[:5]])
        statements = [g for g in self.groups if _IMPACT_STATEMENT_RE.search(g.template) and
                      (g.level in ERROR_LEVELS or g.tags & {"INCIDENT_DECLARED", "ERROR_RATE"})]
        if statements:
            lines.append("Impact statements in the logs: " + "; ".join(self._fmt(self._first_event(g)) for g in statements[:4]) + ".")
            self._claim("Impact statements", ClaimType.OBSERVED, [g.first_event_id for g in statements[:4]])
        first_failure, first_recovery = self._first_failure(), self._first_recovery()
        if first_failure is not None and first_recovery is not None and _seq(first_recovery) > _seq(first_failure):
            window = _gap(first_failure, first_recovery)
            lines.append(f"Failure window: from the first failure [{first_failure.id}] at {first_failure.ts_display} "
                         f"to the first recovery signal [{first_recovery.id}] at {first_recovery.ts_display}"
                         + (f" ({window})." if window else "."))
            self._claim("Failure window", ClaimType.OBSERVED, [first_failure.id, first_recovery.id])
        deps = self._dependency_facts()
        if deps:
            lines.append(deps)
        if not lines:
            return "No ERROR or WARN events from application services were found, so no impact is evident in the logs."
        lines.append("Impact is described only as far as the log events show it; the logs do not quantify affected "
                     "users or transactions unless stated above.")
        return "\n".join(lines)

    def _errors(self) -> str:
        errors = [g for g in self.groups if g.level in ERROR_LEVELS]
        if not errors:
            return "No ERROR or CRITICAL events were found in the evidence."
        lines = [f"{sum(g.count for g in errors)} ERROR/CRITICAL events in {len(errors)} distinct message types (chronological):"]
        for g in errors[:12]:
            event = self._first_event(g)
            suffix = f" (x{g.count})" if g.count > 1 else ""
            lines.append(f"  - {self._fmt(event)}{suffix}")
        self._cite([g.first_event_id for g in errors[:12]])
        self._claim("ERROR events listed", ClaimType.OBSERVED, [g.first_event_id for g in errors[:12]])
        if len(errors) > 12:
            lines.append(f"  ... and {len(errors) - 12} more error message types.")
        exceptions = sorted({g.exception_type.split(".")[-1] for g in errors if g.exception_type})
        if exceptions:
            lines.append("Exception types seen: " + ", ".join(exceptions) + ".")
        return "\n".join(lines)

    def _evidence(self) -> str:
        key = []
        for label, event in (("Earliest abnormal event", self._first_abnormal()),
                             ("First failure", self._first_failure()),
                             ("First recovery signal", self._first_recovery())):
            if event is not None:
                key.append(f"  - {label}: {self._fmt(event)}")
                self._claim(label, ClaimType.OBSERVED, [event.id])
        for g in [g for g in self.groups if "DIAGNOSIS" in g.tags][:3]:
            key.append(f"  - Diagnosis statement: {self._fmt(self._first_event(g))}")
        h = self.analysis.hypothesis
        if h is not None:
            for factor in [f for f in h.factors if f.weight > 0][:5]:
                ids = [i for i in factor.evidence_ids if self.lookup(i) is not None][:3]
                key.append(f"  - Supports the {h.mechanism} explanation: {factor.label} [{', '.join(ids)}]")
                self._cite(ids)
        stats = self.pack.stats
        lines = [
            f"{stats.get('events', 0)} log events were parsed into {stats.get('groups', 0)} distinct message types; "
            f"{len(self.items)} complete events were selected as evidence for this question.",
            "Key evidence:",
            *key,
        ]
        if h is not None:
            lines.append(f"Confidence in the {h.mechanism} explanation: {self.analysis.confidence_explanation()}")
        return "\n".join(lines)

    def _timeline(self) -> str:
        lines = ["Incident timeline (from the log evidence, in time order):"]
        for phase in _PHASE_ORDER:
            events = self._events_in_phase(phase)[:3]
            if not events:
                continue
            lines.append(f"{phase.value}:")
            lines.extend(f"  - {self._fmt(e)}" for e in events)
            self._claim(f"{phase.value} events", ClaimType.OBSERVED, [e.id for e in events])
        start, end = self._first_abnormal(), self._first_recovery() or self._last_abnormal()
        if start is not None and end is not None and start.id != end.id:
            span = _gap(start, end)
            if span:
                lines.append(f"From the first abnormal event [{start.id}] to [{end.id}]: {span}.")
        lines.append(self._causation_caveat())
        return "\n".join(lines)

    def _generic(self) -> str:
        terms = {w for w in re.findall(r"[a-zA-Z][\w\-]{2,}", self.question.lower())}
        matching = [i for i in self.items if terms and any(t in i.event.raw.lower() for t in terms)]
        ranked = sorted(matching or [i for i in self.items if i.event.is_abnormal] or self.items, key=lambda i: -i.score)[:5]
        lines = ["The most relevant evidence for this question:"]
        lines.extend(f"  - {self._fmt(i.event)}" for i in sorted(ranked, key=lambda i: _seq(i.event)))
        self._claim("Relevant events", ClaimType.OBSERVED, [i.event.id for i in ranked])
        overview = self._sequence_sentence()
        if overview:
            lines.append("Overall: " + overview)
        return "\n".join(lines)

    def _documents_only(self) -> str:
        lines = ["No parsed log events are available; relevant document excerpts:"]
        for ev in self.pack.document_evidence[:3]:
            first_lines = "\n    ".join(ev.text.strip().split("\n")[:2])
            lines.append(f"  - [{ev.event_id}] ({ev.source_document}) {first_lines}")
            self._cite([ev.event_id])
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Building blocks
    # ------------------------------------------------------------------ #
    def _sequence_sentence(self) -> str:
        anomaly = self._first_abnormal()
        if anomaly is None:
            return "No WARN or ERROR events were found in the evidence."
        parts = [f"The earliest abnormal event was {self._fmt(anomaly)}"]
        self._claim("Earliest abnormal event", ClaimType.OBSERVED, [anomaly.id])
        degradation = self._first_in_phase(TimelinePhase.DEGRADATION)
        if degradation is not None and degradation.id != anomaly.id:
            parts.append(f"followed by {self._fmt(degradation)}")
            self._claim("Degradation followed", ClaimType.OBSERVED, [degradation.id])
        failure = self._first_failure()
        if failure is not None and failure.id != anomaly.id:
            parts.append(f"and the first failure {self._fmt(failure)}")
            self._claim("First failure", ClaimType.OBSERVED, [failure.id])
        sentence = ", ".join(parts) + "."
        propagated = []
        for event in self._events_in_phase(TimelinePhase.PROPAGATION):
            if event.service and event.service not in [e.service for e in propagated]:
                propagated.append(event)
        if propagated:
            sentence += " Errors then appeared in " + ", ".join(f"{e.service} [{e.id}]" for e in propagated[:5]) + "."
            self._claim("Errors spread to other services", ClaimType.OBSERVED, [e.id for e in propagated[:5]])
        return sentence

    def _diagnosis_statements(self) -> str:
        diagnosis = [self._first_event(g) for g in self.groups if "DIAGNOSIS" in g.tags]
        diagnosis = [e for e in diagnosis if e is not None][:4]
        if not diagnosis:
            return ""
        self._claim("The logs contain explicit diagnosis statements", ClaimType.OBSERVED, [e.id for e in diagnosis])
        return "Diagnosis statements written in the logs: " + "; ".join(self._fmt(e) for e in diagnosis) + "."

    def _explicit_causal_facts(self) -> str:
        causal = [r for r in self.graph_facts if r.is_causal and self._valid_ids(r.evidence_event_ids)]
        if not causal:
            return ""
        parts = [f"{r.source} {r.type.value} {r.target} [{', '.join(self._valid_ids(r.evidence_event_ids)[:2])}]" for r in causal[:4]]
        self._claim("Causal links explicitly stated in the logs", ClaimType.OBSERVED,
                    [i for r in causal[:4] for i in self._valid_ids(r.evidence_event_ids)[:2]])
        return "Causal links stated explicitly in the logs: " + "; ".join(parts) + "."

    def _dependency_facts(self) -> str:
        deps = [r for r in self.graph_facts
                if r.type in (RelationType.DEPENDS_ON, RelationType.AFFECTS) and r.basis == "explicit_text"
                and self._valid_ids(r.evidence_event_ids)]
        deps = [r for r in deps if r.type == RelationType.DEPENDS_ON][:4]
        if not deps:
            return ""
        parts = [f"{r.source} depends on {r.target} [{self._valid_ids(r.evidence_event_ids)[0]}]" for r in deps]
        self._claim("Dependencies stated in the logs", ClaimType.OBSERVED, [self._valid_ids(r.evidence_event_ids)[0] for r in deps])
        return "Dependencies stated in the logs: " + "; ".join(parts) + "."

    def _report_root_cause(self) -> str:
        report = self.report
        if not report or report.root_cause.source != "gemini":
            return ""
        rc = report.root_cause
        ids = self._valid_ids(rc.root_cause_evidence_ids)
        if rc.root_cause_type not in (ClaimType.CONFIRMED, ClaimType.LIKELY) or not ids:
            return ""
        self._claim(rc.root_cause, rc.root_cause_type, ids)
        return (f"The investigation report assessed the root cause as {rc.root_cause_type.value}: "
                f"{rc.root_cause} [{', '.join(ids[:4])}].")

    def _causation_caveat(self) -> str:
        has_diagnosis = any("DIAGNOSIS" in g.tags for g in self.groups)
        has_causal = any(r.is_causal for r in self.graph_facts)
        if has_diagnosis or has_causal:
            self._claim("Root cause beyond the cited statements", ClaimType.LIKELY,
                        [g.first_event_id for g in self.groups if "DIAGNOSIS" in g.tags][:2])
            return ("What this establishes: the order of events (OBSERVED) plus the cause statements cited above. "
                    "This deterministic summary cannot verify that link further, so treat it as LIKELY, not CONFIRMED.")
        self._claim("Root cause", ClaimType.UNKNOWN, [])
        return ("What this establishes: the order of events (temporal correlation). The available evidence does not "
                "conclusively establish which event caused the outage (UNKNOWN).")

    # ------------------------------------------------------------------ #
    def _phases(self) -> Dict[str, TimelinePhase]:
        items = self.items
        first_abnormal = next((i.event for i in items if i.event.level in ("WARN", "ERROR", "CRITICAL")), None)
        first_error = next((i.event for i in items if i.event.level in ERROR_LEVELS), None)
        first_warn_group = next((i.group.id for i in items if i.event.level == "WARN"), None)
        return {
            i.event.id: TimelineBuilder.phase_for(i.event, i.group, first_abnormal, first_error, first_warn_group)
            for i in items
        }

    def _events_in_phase(self, phase: TimelinePhase) -> List[LogEvent]:
        seen_groups, out = set(), []
        for item in self.items:
            if self.phase.get(item.event.id) == phase and item.group.id not in seen_groups:
                seen_groups.add(item.group.id)
                out.append(item.event)
        return out

    def _first_in_phase(self, phase: TimelinePhase) -> Optional[LogEvent]:
        events = self._events_in_phase(phase)
        return events[0] if events else None

    def _first_abnormal(self) -> Optional[LogEvent]:
        return next((i.event for i in self.items if i.event.level in ("WARN", "ERROR", "CRITICAL")), None)

    def _first_failure(self) -> Optional[LogEvent]:
        return self._first_in_phase(TimelinePhase.FAILURE) or next(
            (i.event for i in self.items if i.event.level in ERROR_LEVELS), None)

    def _first_recovery(self) -> Optional[LogEvent]:
        return self._first_in_phase(TimelinePhase.RECOVERY)

    def _last_abnormal(self) -> Optional[LogEvent]:
        return next((i.event for i in reversed(self.items) if i.event.level in ("WARN", "ERROR", "CRITICAL")), None)

    def _first_event(self, group: EventGroup) -> Optional[LogEvent]:
        return self.lookup(group.first_event_id)

    def _valid_ids(self, ids: Sequence[str]) -> List[str]:
        return [i for i in ids if self.lookup(i) is not None]

    def _fmt(self, event: Optional[LogEvent]) -> str:
        if event is None:
            return "(event unavailable)"
        self._cite([event.id])
        service = f"{event.service} " if event.service else ""
        return f"[{event.id}] {event.ts_display} {service}{event.level or ''}: {event.message}"

    def _cite_text(self, text: str) -> None:
        self._cite([m.group(0) for m in re.finditer(r"\bE\d{5,}\b", text)])

    def _cite(self, ids: Sequence[str]) -> None:
        for eid in ids:
            if eid and eid not in self.cited and (self.lookup(eid) is not None or eid in self.pack.valid_event_ids):
                self.cited.append(eid)

    def _claim(self, text: str, claim_type: ClaimType, ids: Sequence[str]) -> None:
        valid = self._valid_ids(ids)
        if claim_type in (ClaimType.OBSERVED, ClaimType.CONFIRMED) and not valid:
            claim_type = ClaimType.UNKNOWN
        self.claims.append(Claim(text=text, type=claim_type, evidence_ids=valid, citations_valid=bool(valid)))
        self._cite(valid)


def _seq(event: Optional[LogEvent]) -> tuple:
    if event is None:
        return (1, datetime.max, 0)
    return (0, event.ts or datetime.max, event.seq)


def _gap(a: LogEvent, b: LogEvent) -> str:
    if not a.ts or not b.ts or b.ts < a.ts:
        return ""
    seconds = int((b.ts - a.ts).total_seconds())
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"
