"""
Query-aware evidence selection under a hard token budget.

Input : the investigation's event store (every LogEvent + EventGroup).
Output: an `EvidencePack` - complete events, rendered chronologically with
        citable ids, whose estimated size never exceeds `budget_tokens`.

Selection
  1. Score every group (severity, RCA signal tags, rarity, incident window,
     question relevance from Qdrant hits, entity / id match).
  2. Protected tiers, added first and in order:
       T0 (chat) events that literally match ids/terms in the question
       T1 first occurrence of every ERROR/CRITICAL template
       T2 earliest abnormal event, first occurrence of each WARN template
          before the first ERROR, deployments/config changes just before it
       T3 diagnosis statements, incident declaration, first recovery, last
          abnormal event
       T4 distinct occurrences hidden by dedup: later bursts, distinct
          request/trace ids, metric peaks, last occurrence
  3. Fill the rest by score: remaining abnormal templates, causal
     predecessors (same request id / same service just before an error),
     recovery and baseline context (INFO capped at 25% of the budget).

Guarantees
  * Events are rendered whole. When the budget is tight a multiline event may
    be rendered COMPACT (header + exception lines + top frames) or HEADER-only;
    every line shown is a complete original line and omissions are stated.
  * Repeated templates appear once with an annotation (count, time span,
    first/last ids, bursts, metric trend), never as N copies.
  * Anything protected that does not fit is listed in `omitted`.
  * `tokens_used <= budget_tokens` always (asserted).
"""
from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.config import get_settings
from app.core.metrics import current_metrics
from app.models.schemas import Evidence, RetrievedChunk
from app.services.ingestion.events import ERROR_LEVELS, EventGroup, LogEvent
from app.services.reasoning.token_budget import TokenBudget, estimate_tokens

_SEVERITY = {"CRITICAL": 100, "ERROR": 80, "WARN": 40, "INFO": 5, "DEBUG": 0, "TRACE": 0}
_STRONG_TAGS = {"TIMEOUT", "HTTP_5XX", "DB", "POOL", "CIRCUIT_BREAKER", "RETRY_EXHAUSTED",
                "DEPENDENCY_UNAVAILABLE", "LEAK", "STARVATION"}
_MEDIUM_TAGS = {"RETRY", "LATENCY", "ERROR_RATE"}
_CONTEXT_TAGS = {"INCIDENT_DECLARED", "DIAGNOSIS", "DEPLOYMENT", "CONFIG_CHANGE", "REMEDIATION"}
_RECOVERY_TAGS = {"RECOVERY", "INCIDENT_RESOLVED"}
_OMISSION_RESERVE_TOKENS = 150
_MIN_OMISSION_RESERVE_TOKENS = 40
_MIN_TS = datetime.min
_INFO_SHARE = 0.25
_PREDECESSOR_LOOKBACK_EVENTS = 200
_PREDECESSOR_WINDOW = timedelta(seconds=60)
_ID_IN_QUESTION_RE = re.compile(r"\b(?:req|txn|trace|ord|sku|cust)[-_]?[\w-]*\d[\w-]*\b|\b[0-9a-f]{8,}\b", re.I)
_STOP = {"the", "and", "what", "which", "why", "did", "was", "were", "how", "when", "first", "this", "that",
         "service", "services", "show", "with", "from", "for", "root", "cause", "incident", "happen", "happened",
         "failed", "fail", "evidence", "there", "any", "does", "about", "unfold", "of", "is", "are", "it"}

FULL, COMPACT, HEADER = "full", "compact", "header"


@dataclass
class SelectedItem:
    event: LogEvent
    group: EventGroup
    tier: str
    score: float
    mode: str
    text: str
    tokens: int
    annotation: str = ""


@dataclass
class EvidencePack:
    items: List[SelectedItem] = field(default_factory=list)
    rendered: str = ""
    tokens_used: int = 0
    budget: int = 0
    omitted: List[str] = field(default_factory=list)
    valid_event_ids: Set[str] = field(default_factory=set)
    stats: Dict[str, int] = field(default_factory=dict)
    document_evidence: List[Evidence] = field(default_factory=list)  # non-log sources (runbooks, PDFs)

    @property
    def protected_count(self) -> int:
        return sum(1 for i in self.items if i.tier != "fill")

    def to_evidence(self, group_point_ids: Dict[str, str], document_names: Dict[str, str]) -> List[Evidence]:
        evidence = []
        max_score = max((i.score for i in self.items), default=1.0) or 1.0
        for item in self.items:
            e, g = item.event, item.group
            evidence.append(
                Evidence(
                    text=item.text,
                    source_document=document_names.get(e.document_id, e.document_id),
                    chunk_id=group_point_ids.get(g.id, g.id),
                    relevance=round(max(item.score, 0.0) / max_score, 3),
                    event_id=e.id,
                    group_id=g.id,
                    timestamp=e.ts_raw,
                    level=e.level,
                    service=e.service,
                    occurrences=g.count,
                    first_ts=g.first_ts.isoformat(sep=" ") if g.first_ts else None,
                    last_ts=g.last_ts.isoformat(sep=" ") if g.last_ts else None,
                    protected=item.tier != "fill",
                )
            )
        return evidence + list(self.document_evidence)

    def append_documents(self, hits: Sequence[RetrievedChunk], document_names: Dict[str, str], budget_tokens: int) -> None:
        """Adds whole non-log chunks (runbooks, reports) while the budget allows.
        They are cited as D##### ids."""
        doc_hits = [h for h in hits or [] if (h.chunk.metadata or {}).get("kind") != "event_group"]
        if not doc_hits:
            return
        budget = TokenBudget(budget_tokens)
        budget.used = self.tokens_used
        blocks = [self.rendered] if self.rendered else []
        for n, hit in enumerate(sorted(doc_hits, key=lambda h: -h.score), start=1):
            doc_id = f"D{n:05d}"
            source = document_names.get(hit.chunk.document_id, hit.chunk.document_id)
            block = f"[{doc_id}] (document: {source})\n{hit.chunk.text}"
            if not budget.add(block):
                continue
            blocks.append(block)
            self.valid_event_ids.add(doc_id)
            self.document_evidence.append(
                Evidence(text=hit.chunk.text, source_document=source, chunk_id=hit.chunk.id,
                         relevance=round(float(hit.score), 3), event_id=doc_id)
            )
        self.rendered = "\n".join(blocks)
        self.tokens_used = estimate_tokens(self.rendered)
        assert self.tokens_used <= budget_tokens, "evidence budget exceeded"


class EvidenceSelector:
    def __init__(self):
        self.settings = get_settings()

    # ------------------------------------------------------------------ #
    def select(
        self,
        events: Sequence[LogEvent],
        groups: Sequence[EventGroup],
        question: str,
        *,
        budget_tokens: int,
        mode: str = "investigation",
        vector_hits: Optional[Sequence[RetrievedChunk]] = None,
        question_matches: Optional[Sequence[LogEvent]] = None,
    ) -> EvidencePack:
        pack = EvidencePack(budget=budget_tokens)
        if not events or not groups or budget_tokens <= 0:
            pack.stats = {"events": len(events), "groups": len(groups), "selected": 0}
            return pack

        if len({e.document_id for e in events}) > 1:  # several files: order by time
            events = sorted(events, key=lambda e: (e.ts is not None, e.ts or _MIN_TS, e.seq))
        events_by_id = {e.id: e for e in events}
        index_of = {e.id: i for i, e in enumerate(events)}
        group_by_id = {g.id: g for g in groups}
        group_pos = {g.id: sorted((index_of[eid], eid) for eid in g.event_ids if eid in index_of) for g in groups}
        groups = [g for g in groups if group_pos[g.id]]
        ctx = _IncidentContext.build(events, index_of, group_pos, self.settings.PRECURSOR_WINDOW_SECONDS)

        relevance = self._relevance(vector_hits, mode)
        question_terms = self._question_terms(question)
        scores = {g.id: self._score(g, ctx, relevance.get(g.id, 0.0), question_terms) for g in groups}

        reserve = min(_OMISSION_RESERVE_TOKENS, max(_MIN_OMISSION_RESERVE_TOKENS, budget_tokens // 10), budget_tokens // 2)
        budget = TokenBudget(budget_tokens - reserve)
        info_budget = int((budget_tokens - reserve) * _INFO_SHARE)
        info_used = 0
        chosen: Dict[str, SelectedItem] = {}
        annotated: Set[str] = set()
        omitted_protected: List[str] = []

        def try_add(event: Optional[LogEvent], tier: str, protected: bool) -> bool:
            nonlocal info_used
            if event is None or event.id in chosen:
                return event is not None
            group = group_by_id[event.group_id]
            is_info = group.level not in ("WARN", "ERROR", "CRITICAL") and not (group.tags & (_RECOVERY_TAGS | _CONTEXT_TAGS))
            modes = (FULL, COMPACT, HEADER) if protected else (FULL, COMPACT)
            for mode_ in modes:
                text = render_event(event, mode_, self.settings.MAX_STACK_FRAMES_KEPT)
                if mode_ != FULL and text == render_event(event, FULL, self.settings.MAX_STACK_FRAMES_KEPT):
                    continue
                annotation = ""
                if group.id not in annotated and group.count > 1:
                    annotation = annotate_group(group, events_by_id)
                cost = budget.cost(text) + (budget.cost(annotation) if annotation else 0)
                if is_info and tier == "fill" and info_used + cost > info_budget:
                    return False
                if budget.used + cost > budget.limit:
                    if annotation and budget.fits(text):
                        annotation, cost = "", budget.cost(text)
                    else:
                        continue
                budget.add(text)
                if annotation:
                    budget.add(annotation)
                    annotated.add(group.id)
                if is_info and tier == "fill":
                    info_used += cost
                chosen[event.id] = SelectedItem(event, group, tier, scores[group.id], mode_, text, cost, annotation)
                return True
            if protected and event.id not in omitted_protected:
                omitted_protected.append(event.id)
            return False

        # ---- T0: literal matches from the question (chat) ----
        for event in list(question_matches or [])[:10]:
            if event.id in events_by_id:
                try_add(event, "T0", True)

        # ---- T1: first occurrence of every ERROR/CRITICAL template ----
        error_groups = sorted(
            (g for g in groups if g.level in ERROR_LEVELS), key=lambda g: (-scores[g.id], group_pos[g.id][0][0])
        )
        for g in error_groups:
            try_add(events_by_id.get(g.first_event_id), "T1", True)

        # ---- T2: precursors ----
        if ctx.first_abnormal is not None:
            try_add(ctx.first_abnormal, "T2", True)
        first_error_pos = index_of[ctx.first_error.id] if ctx.first_error is not None else None
        for g in sorted(groups, key=lambda g: group_pos[g.id][0][0]):
            if g.level == "WARN" and (first_error_pos is None or group_pos[g.id][0][0] < first_error_pos):
                try_add(events_by_id.get(g.first_event_id), "T2", True)
        for g in groups:
            if g.tags & {"DEPLOYMENT", "CONFIG_CHANGE"}:
                first = events_by_id.get(g.first_event_id)
                if first and ctx.in_precursor_window(first):
                    try_add(first, "T2", True)

        # ---- T3: diagnosis / leak / starvation findings, declaration, remediation, first recovery,
        #          last abnormal (root-cause statements must never lose out on vector similarity) ----
        for g in sorted(groups, key=lambda g: group_pos[g.id][0][0]):
            if g.tags & {"DIAGNOSIS", "INCIDENT_DECLARED", "LEAK", "STARVATION"} and ctx.in_incident_span(
                events_by_id[g.first_event_id]
            ):
                try_add(events_by_id.get(g.first_event_id), "T3", True)
        if ctx.first_recovery is not None:
            try_add(ctx.first_recovery, "T3", True)
        if ctx.last_abnormal is not None:
            try_add(ctx.last_abnormal, "T3", True)
        for g in sorted(groups, key=lambda g: group_pos[g.id][0][0]):
            if "REMEDIATION" in g.tags and g.level not in ERROR_LEVELS and ctx.in_incident_span(
                events_by_id[g.first_event_id]
            ):
                try_add(events_by_id.get(g.first_event_id), "T3", True)

        # ---- T4: distinct occurrences hidden by deduplication ----
        abnormal_groups = sorted((g for g in groups if g.level in ("WARN", "ERROR", "CRITICAL")), key=lambda g: -scores[g.id])
        cap = max(1, self.settings.MAX_INSTANCES_PER_GROUP)
        for g in abnormal_groups:
            for eid in g.instance_ids(cap):
                try_add(events_by_id.get(eid), "T4", True)

        # ---- Fill by score ----
        fill: List[Tuple[float, int, LogEvent]] = []
        for event in self._predecessors(chosen.values(), events, index_of):
            fill.append((50.0, index_of[event.id], event))
        for g in groups:
            if g.level in ("DEBUG", "TRACE") and relevance.get(g.id, 0.0) < 0.5:
                continue
            candidate = ctx.representative(g, events_by_id)
            if candidate is not None:
                fill.append((scores[g.id], index_of[candidate.id], candidate))
        recovery_groups = [g for g in groups if g.tags & _RECOVERY_TAGS and g.level not in ERROR_LEVELS]
        for g in recovery_groups:
            fill.append((scores[g.id] + 5, group_pos[g.id][0][0], events_by_id[g.first_event_id]))
        fill.sort(key=lambda t: (-t[0], t[1]))
        for _, _, event in fill:
            if budget.remaining < 8:
                break
            try_add(event, "fill", False)

        # ---- Omissions summary (inside the reserved slice): counts + as many ids as fit ----
        omission_lines: List[str] = []
        remaining = budget_tokens - budget.used

        def block_cost(lines: List[str]) -> int:
            return estimate_tokens("[Not shown]\n" + "\n".join(f"- {l}" for l in lines)) if lines else 0

        missing = [eid for eid in omitted_protected if eid not in chosen]
        if missing:
            prefix = f"{len(missing)} protected events did not fit the evidence budget"
            line = prefix
            for eid in missing:
                candidate = f"{prefix}: {eid}" if line == prefix else f"{line}, {eid}"
                if block_cost([candidate]) > remaining:
                    break
                line = candidate
            if block_cost([line]) <= remaining:
                omission_lines.append(line)
        shown_groups = {item.group.id for item in chosen.values()}
        hidden_abnormal = [g for g in groups if g.level in ("WARN", "ERROR", "CRITICAL") and g.id not in shown_groups]
        if hidden_abnormal:
            hidden_events = sum(g.count for g in hidden_abnormal)
            line = f"{len(hidden_abnormal)} abnormal templates ({hidden_events} events) not shown"
            if block_cost(omission_lines + [line]) <= remaining:
                omission_lines.append(line)

        # ---- Render chronologically ----
        ordered = sorted(chosen.values(), key=lambda item: index_of[item.event.id])
        blocks: List[str] = []
        valid: Set[str] = set()
        for item in ordered:
            blocks.append(item.text)
            valid.add(item.event.id)
            if item.annotation:
                blocks.append(item.annotation)
                valid.update(_ANNOTATION_ID_RE.findall(item.annotation))
        if omission_lines:
            blocks.append("[Not shown]\n" + "\n".join(f"- {line}" for line in omission_lines))
        pack.rendered = "\n".join(blocks)
        pack.items = ordered
        pack.tokens_used = estimate_tokens(pack.rendered)
        pack.omitted = omission_lines
        pack.valid_event_ids = valid
        # Hard guarantee. The per-item accounting is conservative, so this only
        # trims if the joined text estimate disagrees; protected items go last.
        while pack.tokens_used > budget_tokens and pack.items:
            drop = min(pack.items, key=lambda it: (it.tier != "fill", it.score))
            pack.items.remove(drop)
            pack.rendered = "\n".join(
                [x for it in pack.items for x in ([it.text] + ([it.annotation] if it.annotation else []))]
            )
            pack.valid_event_ids = {it.event.id for it in pack.items} | {
                eid for it in pack.items for eid in _ANNOTATION_ID_RE.findall(it.annotation or "")
            }
            pack.tokens_used = estimate_tokens(pack.rendered)
        assert pack.tokens_used <= budget_tokens, "evidence budget exceeded"

        pack.stats = {
            "events": len(events),
            "groups": len(groups),
            "selected": len(pack.items),
            "protected": pack.protected_count,
            "omitted_events": len(events) - len(pack.items),
            "tokens_used": pack.tokens_used,
            "budget": budget_tokens,
        }
        metrics = current_metrics()
        if metrics is not None:
            metrics.incr("selected_events", len(pack.items))
            metrics.incr("protected_events", pack.protected_count)
            metrics.incr("omitted_events", len(events) - len(pack.items))
            metrics.incr("evidence_tokens_used", pack.tokens_used)
            metrics.incr("evidence_budget_tokens", budget_tokens)
        return pack

    # ------------------------------------------------------------------ #
    def _score(self, g: EventGroup, ctx: "_IncidentContext", relevance: float, question_terms: Set[str]) -> float:
        score = float(_SEVERITY.get(g.level or "INFO", 5))
        if "EXCEPTION" in g.tags or g.has_stack_trace:
            score += 30
        if g.tags & _STRONG_TAGS:
            score += 25
        if g.tags & _MEDIUM_TAGS:
            score += 15
        if g.tags & _RECOVERY_TAGS:
            score += 40
        if g.tags & {"INCIDENT_DECLARED", "DIAGNOSIS", "DEPLOYMENT"}:
            score += 20
        score += 20.0 / (1.0 + math.log2(max(g.count, 1)))
        if ctx.overlaps_window(g):
            score += 15
        elif g.level in ("INFO", "DEBUG", "TRACE", None):
            score -= 10
        score += relevance
        if question_terms:
            haystack = f"{g.service or ''} {g.template}".lower()
            if any(t in haystack for t in question_terms):
                score += 25
        return score

    def _relevance(self, hits: Optional[Sequence[RetrievedChunk]], mode: str) -> Dict[str, float]:
        weight = 40.0 if mode == "chat" else 20.0
        out: Dict[str, float] = {}
        for hit in hits or []:
            gid = (hit.chunk.metadata or {}).get("group_id")
            if gid:
                out[gid] = max(out.get(gid, 0.0), weight * max(0.0, float(hit.score)))
        return out

    @staticmethod
    def _question_terms(question: str) -> Set[str]:
        words = re.findall(r"[a-zA-Z][\w\-]{2,}", (question or "").lower())
        return {w for w in words if w not in _STOP}

    @staticmethod
    def question_ids(question: str) -> List[str]:
        return _ID_IN_QUESTION_RE.findall(question or "")

    @staticmethod
    def _predecessors(selected: Iterable[SelectedItem], events: Sequence[LogEvent], index_of: Dict[str, int]) -> List[LogEvent]:
        """Up to two events of the same service (or sharing a request id) just
        before each selected first ERROR occurrence, within 60 s."""
        out: List[LogEvent] = []
        for item in list(selected):
            event = item.event
            if event.level not in ERROR_LEVELS or item.tier != "T1":
                continue
            idx = index_of.get(event.id)
            if idx is None:
                continue
            req = (event.ids or {}).get("request_id")
            found = 0
            for back in range(idx - 1, max(-1, idx - _PREDECESSOR_LOOKBACK_EVENTS), -1):
                prev = events[back]
                if event.ts and prev.ts and event.ts - prev.ts > _PREDECESSOR_WINDOW:
                    break
                same_req = req is not None and (prev.ids or {}).get("request_id") == req
                if same_req or (prev.service and prev.service == event.service):
                    out.append(prev)
                    found += 1
                    if found >= 2:
                        break
        return out


# --------------------------------------------------------------------------- #
# Incident context
# --------------------------------------------------------------------------- #
@dataclass
class _IncidentContext:
    """Incident boundaries in chronological *positions* of the event list."""

    first_abnormal: Optional[LogEvent]
    first_error: Optional[LogEvent]
    last_abnormal: Optional[LogEvent]
    first_recovery: Optional[LogEvent]
    window_start: int
    window_end: int
    precursor_start: int
    index_of: Dict[str, int]
    group_pos: Dict[str, List[Tuple[int, str]]]

    @classmethod
    def build(
        cls,
        events: Sequence[LogEvent],
        index_of: Dict[str, int],
        group_pos: Dict[str, List[Tuple[int, str]]],
        precursor_seconds: int,
    ) -> "_IncidentContext":
        abnormal = ("WARN", "ERROR", "CRITICAL")
        first_abnormal = next((e for e in events if e.level in abnormal), None)
        first_error = next((e for e in events if e.level in ERROR_LEVELS), None)
        last_abnormal = next((e for e in reversed(events) if e.level in abnormal), None)
        if first_abnormal is None:
            return cls(None, None, None, None, 0, len(events) - 1, 0, index_of, group_pos)

        start = index_of[first_abnormal.id]
        end = index_of[last_abnormal.id]
        first_recovery = next(
            (e for e in events[start:] if e.tags & _RECOVERY_TAGS and e.level not in ERROR_LEVELS), None
        )
        # Precursor window: up to `precursor_seconds` before the first abnormal event.
        limit = timedelta(seconds=precursor_seconds)
        i = start
        while i > 0:
            prev = events[i - 1]
            if first_abnormal.ts and prev.ts:
                if first_abnormal.ts - prev.ts > limit:
                    break
            elif start - i >= 20:
                break
            i -= 1
        # Recovery tail: recovery-tagged events after the last abnormal one.
        tail_end = end
        for j in range(end + 1, min(len(events), end + 5000)):
            if events[j].tags & _RECOVERY_TAGS:
                tail_end = j
        return cls(first_abnormal, first_error, last_abnormal, first_recovery, start, tail_end, i, index_of, group_pos)

    def in_precursor_window(self, event: LogEvent) -> bool:
        return self.precursor_start <= self.index_of[event.id] <= self.window_start

    def in_incident_span(self, event: LogEvent) -> bool:
        return self.precursor_start <= self.index_of[event.id] <= self.window_end + 50

    def overlaps_window(self, g: EventGroup) -> bool:
        positions = self.group_pos.get(g.id) or []
        idx = bisect.bisect_left(positions, (self.precursor_start, ""))
        return idx < len(positions) and positions[idx][0] <= self.window_end

    def representative(self, g: EventGroup, events_by_id: Dict[str, LogEvent]) -> Optional[LogEvent]:
        """For abnormal/context groups the first occurrence; for INFO/DEBUG groups
        the occurrence closest to (just before) the incident, i.e. baseline context."""
        positions = self.group_pos.get(g.id) or []
        if not positions:
            return None
        if g.level in ("WARN", "ERROR", "CRITICAL") or g.tags & (_RECOVERY_TAGS | _CONTEXT_TAGS):
            return events_by_id.get(positions[0][1])
        if not self.overlaps_window(g):
            return None
        idx = bisect.bisect_left(positions, (self.window_start, ""))
        if idx > 0 and positions[idx - 1][0] >= self.precursor_start:
            idx -= 1
        idx = min(idx, len(positions) - 1)
        return events_by_id.get(positions[idx][1])


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_ANNOTATION_ID_RE = re.compile(r"\bE\d{5,}\b")


def render_event(event: LogEvent, mode: str = FULL, max_frames: int = 8) -> str:
    """Renders a whole event; every emitted line is a complete original line."""
    lines = event.raw.split("\n")
    head = f"[{event.id}] {lines[0]}"
    rest = lines[1:]
    if not rest or mode == FULL:
        return "\n".join([head] + [f"    {l.strip()}" for l in rest])
    if mode == HEADER:
        return f"{head}\n    [{len(rest)} continuation lines omitted]"
    kept: List[str] = []
    frames = 0
    for line in rest:
        stripped = line.strip()
        if stripped.startswith("at "):
            if frames < max_frames:
                kept.append(stripped)
                frames += 1
        elif stripped.startswith("Caused by") or "Exception" in stripped or "Error" in stripped or stripped.startswith("..."):
            kept.append(stripped)
    omitted = len(rest) - len(kept)
    body = [f"    {l}" for l in kept]
    if omitted:
        body.append(f"    [{omitted} stack lines omitted]")
    return "\n".join([head] + body)


def annotate_group(group: EventGroup, events_by_id: Dict[str, LogEvent]) -> str:
    first = events_by_id.get(group.first_event_id)
    last = events_by_id.get(group.last_event_id)
    parts = [f"same template x{group.count} total"]
    if first and last and first.id != last.id:
        parts.append(f"first {first.id} {first.ts_display}, last {last.id} {last.ts_display}")
    if len(group.bursts) > 1:
        starts = ", ".join(f"{b.start_event_id}(x{b.count})" for b in group.bursts[:4])
        parts.append(f"{len(group.bursts)} separate bursts: {starts}")
    if group.peak_event_id and group.peak_event_id not in (group.first_event_id,):
        parts.append(f"peak {group.peak_event_id}")
    trend = group.metric_summary()
    if trend:
        parts.append(trend)
    if len(group.distinct_request_ids) > 1:
        ids = list(group.distinct_request_ids.items())[:4]
        parts.append("distinct ids: " + ", ".join(f"{rid}={eid}" for rid, eid in ids))
    return "    ^ " + "; ".join(parts)
