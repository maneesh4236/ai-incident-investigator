"""
Root cause analysis: ONE Gemini reasoning call per investigation.

`reason()` sends the budgeted evidence pack (complete events with citable
ids), the deterministic timeline skeleton and evidence-backed graph facts in
a single request, and asks for root cause, cause chain, typed/cited claims,
timeline phase labels, affected systems, executive summary and
recommendations together - replacing the former three calls (RCA, timeline,
report narrative).

Every response is validated deterministically (`claims.ClaimValidator`):
invented ids are dropped, claim types are downgraded when the citations do
not support them, confidence is coerced and capped. If Gemini is not
configured, fails (429/5xx/timeout/quota/...) or returns unusable output, a
deterministic, evidence-cited fallback is returned and marked degraded.

`analyze()` is the legacy chunk-based entry point, kept for compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.metrics import current_metrics
from app.models.schemas import (
    Claim,
    ClaimType,
    Evidence,
    HybridRetrievalResult,
    RootCauseResult,
    Timeline,
    TimelinePhase,
)
from app.services.graph.graph_builder import GraphBuilder
from app.services.ingestion.events import ERROR_LEVELS, LogEvent
from app.services.llm.gemini_client import GeminiClient
from app.services.reasoning.claims import (
    ClaimValidator,
    cap_confidence,
    coerce_confidence,
    coerce_str,
    coerce_str_list,
    dict_get,
    normalize_id,
)
from app.services.reasoning.evidence_builder import EvidenceBuilder
from app.services.reasoning.evidence_selector import EvidencePack
from app.services.reasoning.token_budget import estimate_tokens

logger = get_logger("reasoning.root_cause_analyzer")

INVESTIGATION_SYSTEM_PROMPT = """You are a Principal Site Reliability Engineer performing an evidence-grounded root cause analysis.

EVIDENCE: complete log events, chronological, each prefixed with an id like [E00042] (documents use [D00001]).
"^ same template xN" lines summarise repeated occurrences and list their ids. Nothing else is known.

RULES
1. Use only the evidence provided. Cite ids for every claim in "evidence_ids". Never cite an id that is not shown.
2. Classify every claim:
   OBSERVED  = directly stated by the cited event(s)
   INFERRED  = reasoned from cited events but not stated
   LIKELY    = best-supported explanation, not proven
   CONFIRMED = a cited event explicitly states the diagnosis/cause ("root cause", "identified", "caused by", "due to")
   UNKNOWN   = evidence is insufficient
3. Correlation is not causation. Events close in time, or a warning that precedes a failure, do not prove one caused the other.
4. The earliest warning is not automatically the root cause.
5. A metric below 100% (e.g. "pool usage 78%") is not exhaustion unless an event explicitly reports exhaustion, timeout or starvation.
6. Use precise vocabulary: warning/anomaly (abnormal signal), degradation (worsening performance), failure (errors / requests failing),
   propagation (failure spreading to dependents), recovery. Do not call something a failure unless an ERROR/failure event supports it.
7. If the evidence cannot establish something, say "Evidence is insufficient to establish this." and use type UNKNOWN.
8. Timeline: you may relabel the phase of skeleton entries (by their event_id) only; do not add events.

Respond ONLY with JSON in exactly this shape:
{
  "root_cause": {"statement": "...", "type": "OBSERVED|INFERRED|LIKELY|CONFIRMED|UNKNOWN", "evidence_ids": ["E00001"]},
  "cause_chain": [{"step": "...", "type": "...", "evidence_ids": ["..."]}],
  "claims": [{"text": "...", "type": "...", "evidence_ids": ["..."]}],
  "timeline": [{"event_id": "E00001", "phase": "PRECURSOR|ANOMALY|DEGRADATION|FAILURE|PROPAGATION|RECOVERY|CONTEXT"}],
  "affected_systems": [{"name": "service-name", "impact": "warning|degraded|failed|unknown", "evidence_ids": ["..."]}],
  "executive_summary": "2-4 factual sentences",
  "recommendations": ["specific, actionable item"],
  "insufficient_evidence": ["what cannot be established and why"],
  "confidence": 0.0
}"""

_PROMPT_SCAFFOLD_TOKENS = 120


@dataclass
class InvestigationResult:
    root_cause: RootCauseResult
    timeline: Timeline
    executive_summary: str
    recommendations: List[str]
    degraded: bool = False
    degradation_reason: Optional[str] = None
    cause_chain_structured: List[dict] = field(default_factory=list)
    rejected_ids: List[str] = field(default_factory=list)
    downgrades: List[str] = field(default_factory=list)
    prompt_tokens_estimated: int = 0


class RootCauseAnalyzer:
    def __init__(
        self,
        graph_builder: GraphBuilder,
        llm_client: GeminiClient | None = None,
        evidence_builder: EvidenceBuilder | None = None,
    ):
        self.graph_builder = graph_builder
        self.llm_client = llm_client or GeminiClient()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.settings = get_settings()

    # ------------------------------------------------------------------ #
    # Budget planning
    # ------------------------------------------------------------------ #
    def evidence_budget_for(self, question: str) -> int:
        s = self.settings
        fixed = (
            estimate_tokens(INVESTIGATION_SYSTEM_PROMPT)
            + estimate_tokens(question)
            + s.MAX_GRAPH_CONTEXT_TOKENS
            + self._skeleton_reserve()
            + _PROMPT_SCAFFOLD_TOKENS
        )
        return max(0, min(s.MAX_GEMINI_EVIDENCE_TOKENS, s.MAX_GEMINI_CONTEXT_TOKENS - fixed))

    def _skeleton_reserve(self) -> int:
        return 12 * self.settings.TIMELINE_MAX_EVENTS

    def build_prompt(self, question: str, pack: EvidencePack, timeline: Timeline, graph_facts: Sequence[str]) -> str:
        s = self.settings
        skeleton = [f"{e.event_ids[0]}={e.phase.value if e.phase else 'CONTEXT'}" for e in timeline.events if e.event_ids]
        skeleton_text = "; ".join(skeleton)
        while skeleton and estimate_tokens(skeleton_text) > self._skeleton_reserve():
            skeleton.pop()
            skeleton_text = "; ".join(skeleton)

        facts: List[str] = []
        used = 0
        for fact in graph_facts:
            cost = estimate_tokens(fact)
            if used + cost > s.MAX_GRAPH_CONTEXT_TOKENS:
                break
            facts.append(fact)
            used += cost

        omitted = ""
        if pack.omitted:
            omitted = "\n(Some evidence was omitted for budget; see the [Not shown] section.)"

        def assemble(facts_lines: List[str], skeleton_line: str) -> str:
            return (
                f"QUESTION: {question}\n\n"
                f"EVIDENCE ({len(pack.items)} events selected from {pack.stats.get('events', 0)} parsed; "
                f"{pack.stats.get('groups', 0)} distinct templates):{omitted}\n"
                f"{pack.rendered}\n\n"
                f"TIMELINE SKELETON (event_id=proposed phase): {skeleton_line or 'none'}\n\n"
                "GRAPH FACTS (relationships stated explicitly in the logs, with evidence ids):\n"
                + ("\n".join(facts_lines) if facts_lines else "none")
            )

        prompt = assemble(facts, skeleton_text)
        # Final hard guard (should never trigger given the reserves): drop graph facts, then skeleton.
        while estimate_tokens(INVESTIGATION_SYSTEM_PROMPT) + estimate_tokens(prompt) > s.MAX_GEMINI_CONTEXT_TOKENS:
            if facts:
                facts.pop()
            elif skeleton:
                skeleton.pop()
                skeleton_text = "; ".join(skeleton)
            else:
                break
            prompt = assemble(facts, skeleton_text)
        return prompt

    # ------------------------------------------------------------------ #
    # Single-call reasoning
    # ------------------------------------------------------------------ #
    def reason(
        self,
        investigation_id: str,
        question: str,
        pack: EvidencePack,
        timeline: Timeline,
        graph_facts: Sequence[str],
        event_lookup: Callable[[str], Optional[LogEvent]],
        evidence: List[Evidence],
    ) -> InvestigationResult:
        prompt = self.build_prompt(question, pack, timeline, graph_facts)
        estimated = estimate_tokens(INVESTIGATION_SYSTEM_PROMPT) + estimate_tokens(prompt)
        metrics = current_metrics()
        if metrics is not None:
            metrics.incr("prompt_tokens_estimated", estimated)
            metrics.incr("context_budget_tokens", self.settings.MAX_GEMINI_CONTEXT_TOKENS)

        if not pack.items and not pack.document_evidence:
            result = self.deterministic_result(investigation_id, pack, timeline, evidence, "no_evidence")
            result.prompt_tokens_estimated = 0
            return result

        llm = self.llm_client.call(
            prompt,
            INVESTIGATION_SYSTEM_PROMPT,
            purpose="investigation",
            json_mode=True,
            max_output_tokens=self.settings.GEMINI_MAX_OUTPUT_TOKENS_INVESTIGATION,
        )
        if not llm.ok:
            logger.warning(f"Investigation Gemini call unavailable ({llm.error_kind}); using deterministic fallback")
            result = self.deterministic_result(investigation_id, pack, timeline, evidence, llm.error_kind or "gemini_error")
            result.prompt_tokens_estimated = estimated
            return result

        parsed = self._parse(investigation_id, llm.data, pack, timeline, event_lookup, evidence)
        if parsed is None:
            if metrics is not None:
                metrics.record_error("invalid_llm_output")
            result = self.deterministic_result(investigation_id, pack, timeline, evidence, "invalid_llm_output")
            result.prompt_tokens_estimated = estimated
            return result
        parsed.prompt_tokens_estimated = estimated
        return parsed

    def _parse(
        self,
        investigation_id: str,
        data: dict,
        pack: EvidencePack,
        timeline: Timeline,
        event_lookup: Callable[[str], Optional[LogEvent]],
        evidence: List[Evidence],
    ) -> Optional[InvestigationResult]:
        validator = ClaimValidator(pack.valid_event_ids, event_lookup)

        raw_root = data.get("root_cause")
        if isinstance(raw_root, dict):
            statement = coerce_str(raw_root, ("statement", "text", "root_cause"))
            root_claim = validator.validate(statement, raw_root.get("type"), raw_root.get("evidence_ids"))
        else:
            statement = coerce_str(raw_root)
            root_claim = validator.validate(statement, data.get("root_cause_type"), data.get("root_cause_evidence_ids"))
        if not statement:
            return None

        chain_claims = validator.validate_items(data.get("cause_chain"), ("step", "text", "statement"))
        claims = validator.validate_items(data.get("claims"))

        overrides: Dict[str, TimelinePhase] = {}
        raw_timeline = data.get("timeline")
        for entry in raw_timeline if isinstance(raw_timeline, list) else []:
            if not isinstance(entry, dict):
                continue
            eid = normalize_id(entry.get("event_id"))
            phase = str(entry.get("phase") or "").upper()
            if eid and phase in TimelinePhase.__members__:
                overrides[eid] = TimelinePhase[phase]
        from app.services.reasoning.timeline_builder import TimelineBuilder

        TimelineBuilder.apply_phase_overrides(timeline, overrides)

        affected: List[str] = []
        raw_affected = data.get("affected_systems")
        for item in raw_affected if isinstance(raw_affected, list) else coerce_str_list(raw_affected):
            name = coerce_str(item, ("name", "service", "system"))
            if name and name not in affected:
                affected.append(name)

        confidence = cap_confidence(coerce_confidence(dict_get(data, "confidence", "confidence_score"), 0.5), root_claim.type)
        summary = coerce_str(data.get("executive_summary"))
        recommendations = coerce_str_list(data.get("recommendations"))
        insufficient = coerce_str_list(data.get("insufficient_evidence"))

        root = RootCauseResult(
            root_cause=statement,
            cause_chain=[c.text for c in chain_claims],
            confidence_score=confidence,
            evidence=evidence,
            affected_systems=affected,
            root_cause_type=root_claim.type,
            root_cause_evidence_ids=root_claim.evidence_ids,
            claims=chain_claims + claims,
            insufficient_evidence=insufficient,
            source="gemini",
        )
        if validator.rejected_ids:
            logger.warning(f"Rejected {len(validator.rejected_ids)} invented/unknown evidence ids: {validator.rejected_ids[:10]}")
        return InvestigationResult(
            root_cause=root,
            timeline=timeline,
            executive_summary=summary or self._deterministic_summary(root, timeline, pack),
            recommendations=recommendations or self._default_recommendations(root.confidence_score),
            cause_chain_structured=[
                {"step": c.text, "type": c.type.value, "evidence_ids": c.evidence_ids} for c in chain_claims
            ],
            rejected_ids=list(validator.rejected_ids),
            downgrades=list(validator.downgrades),
        )

    # ------------------------------------------------------------------ #
    # Deterministic fallback (Gemini unavailable / failed / invalid output)
    # ------------------------------------------------------------------ #
    def deterministic_result(
        self, investigation_id: str, pack: EvidencePack, timeline: Timeline, evidence: List[Evidence], reason: str
    ) -> InvestigationResult:
        items = pack.items
        first_abnormal = next((i.event for i in items if i.event.level in ("WARN", "ERROR", "CRITICAL")), None)
        first_error = next((i.event for i in items if i.event.level in ERROR_LEVELS), None)
        first_recovery = next(
            (e for e in timeline.events if e.phase == TimelinePhase.RECOVERY and e.event_ids), None
        )

        claims: List[Claim] = []
        if first_abnormal is not None:
            claims.append(Claim(text=f"Earliest abnormal event ({first_abnormal.level}) at {first_abnormal.ts_display}: "
                                f"{first_abnormal.message}", type=ClaimType.OBSERVED, evidence_ids=[first_abnormal.id]))
        seen_services = set()
        for item in items:
            e = item.event
            if e.level in ERROR_LEVELS and e.service not in seen_services and len(seen_services) < 6:
                seen_services.add(e.service)
                claims.append(Claim(text=f"First ERROR from {e.service or 'unknown service'} at {e.ts_display}: {e.message}",
                                    type=ClaimType.OBSERVED, evidence_ids=[e.id]))
        if first_recovery is not None:
            claims.append(Claim(text=f"First recovery signal at {first_recovery.timestamp}: {first_recovery.title}",
                                type=ClaimType.OBSERVED, evidence_ids=first_recovery.event_ids[:1]))

        if first_abnormal is None:
            statement = ("No WARN/ERROR events were found in the evidence. Evidence is insufficient to establish "
                         "an incident or its root cause.")
        else:
            statement = ("Evidence is insufficient to establish a root cause automatically "
                         f"(AI reasoning unavailable: {reason}). Earliest abnormal event: [{first_abnormal.id}] "
                         f"{first_abnormal.message}.")
            if first_error is not None:
                statement += f" First ERROR: [{first_error.id}] {first_error.service or ''} {first_error.message}."

        chain: List[str] = []
        chain_struct: List[dict] = []
        for phase in (TimelinePhase.PRECURSOR, TimelinePhase.ANOMALY, TimelinePhase.DEGRADATION,
                      TimelinePhase.FAILURE, TimelinePhase.PROPAGATION, TimelinePhase.RECOVERY):
            entry = next((e for e in timeline.events if e.phase == phase and e.event_ids), None)
            if entry is not None:
                text = f"{phase.value}: [{entry.event_ids[0]}] {entry.title} (observed sequence; causation not established)"
                chain.append(text)
                chain_struct.append({"step": text, "type": ClaimType.OBSERVED.value, "evidence_ids": entry.event_ids[:1]})

        affected = sorted({i.event.service for i in items if i.event.level in ERROR_LEVELS and i.event.service})
        root = RootCauseResult(
            root_cause=statement,
            cause_chain=chain,
            confidence_score=0.2 if first_abnormal is not None else 0.1,
            evidence=evidence,
            affected_systems=affected,
            root_cause_type=ClaimType.UNKNOWN,
            root_cause_evidence_ids=[first_abnormal.id] if first_abnormal is not None else [],
            claims=claims,
            insufficient_evidence=["Root cause could not be reasoned about without the AI reasoning step; "
                                   "the observed sequence is shown instead."],
            source="deterministic",
        )
        return InvestigationResult(
            root_cause=root,
            timeline=timeline,
            executive_summary=self._deterministic_summary(root, timeline, pack),
            recommendations=self._default_recommendations(root.confidence_score)
            + ["Re-run the investigation when the Gemini API is available for an evidence-cited root cause."],
            degraded=True,
            degradation_reason=reason,
            cause_chain_structured=chain_struct,
        )

    @staticmethod
    def _deterministic_summary(root: RootCauseResult, timeline: Timeline, pack: EvidencePack) -> str:
        phases = [e.phase.value for e in timeline.events if e.phase and e.phase != TimelinePhase.CONTEXT]
        ordered: List[str] = []
        for p in phases:
            if p not in ordered:
                ordered.append(p)
        return (
            f"{pack.stats.get('events', 0)} log events were parsed into {pack.stats.get('groups', 0)} distinct templates; "
            f"{len(pack.items)} complete events were selected as evidence. "
            f"Observed phases: {' -> '.join(ordered) or 'none'}. "
            f"Root cause ({root.root_cause_type.value}): {root.root_cause}"
        )

    @staticmethod
    def _default_recommendations(confidence: float) -> List[str]:
        recs = [
            "Add automated alerting on the earliest abnormal signal in the timeline to reduce detection time.",
            "Introduce a runbook step to validate this failure mode during future incident triage.",
        ]
        if confidence < 0.5:
            recs.append("Gather additional logs/traces; current evidence yields a low-confidence root cause.")
        return recs

    # ------------------------------------------------------------------ #
    # Legacy chunk-based API (kept for compatibility; not used by routes)
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        investigation_id: str,
        retrieval: HybridRetrievalResult,
        document_names: dict[str, str] | None = None,
    ) -> RootCauseResult:
        evidence = self.evidence_builder.build(retrieval, document_names)

        if self.llm_client.is_configured:
            result = self._analyze_with_llm(retrieval, evidence)
            if result:
                return result

        logger.info("Falling back to graph-heuristic root cause analysis")
        return self._analyze_heuristic(investigation_id, retrieval, evidence)

    def _analyze_with_llm(self, retrieval: HybridRetrievalResult, evidence: List[Evidence]) -> RootCauseResult | None:
        chunk_text = "\n---\n".join(e.text for e in evidence[:8])
        graph_paths = "; ".join(" -> ".join(p) for p in retrieval.graph_context.paths) or "none found"
        prompt = f"Question: {retrieval.query}\n\nLog/document excerpts:\n{chunk_text}\n\nGraph causal paths: {graph_paths}\n"
        data = self.llm_client.generate_json(prompt, system_instruction=INVESTIGATION_SYSTEM_PROMPT)
        statement = coerce_str(data.get("root_cause"), ("statement", "text")) if data else ""
        if not statement:
            return None
        return RootCauseResult(
            root_cause=statement,
            cause_chain=coerce_str_list(data.get("cause_chain")),
            confidence_score=coerce_confidence(dict_get(data, "confidence_score", "confidence"), 0.6),
            evidence=evidence[:8],
            affected_systems=coerce_str_list(data.get("affected_systems")),
            source="gemini",
        )

    def _analyze_heuristic(
        self, investigation_id: str, retrieval: HybridRetrievalResult, evidence: List[Evidence]
    ) -> RootCauseResult:
        paths = retrieval.graph_context.paths  # causal-only paths
        if paths:
            best_path = max(paths, key=len)
            root_cause = best_path[0]
            cause_chain = best_path
            confidence = min(0.5 + 0.05 * len(best_path), 0.85)
        else:
            ranked = self.graph_builder.centrality_ranked_nodes(investigation_id)
            root_cause = ranked[0] if ranked else "Unknown (insufficient evidence)"
            cause_chain = ranked[:3] if ranked else []
            confidence = 0.35 if ranked else 0.1

        affected = [e.name for e in retrieval.graph_context.entities if e.type.value == "SERVICE"]

        return RootCauseResult(
            root_cause=root_cause,
            cause_chain=cause_chain,
            confidence_score=round(confidence, 2),
            evidence=evidence[:8],
            affected_systems=affected,
        )
