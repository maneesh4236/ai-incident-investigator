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
    strip_unverified_ids,
)
from app.services.reasoning.evidence_builder import EvidenceBuilder
from app.services.reasoning.evidence_selector import EvidencePack
from app.services.reasoning.incident_analysis import (
    ALERTING_SERVICE_RE,
    IncidentAnalysis,
    analyze_incident,
    confidence_label,
)
from app.services.reasoning.token_budget import estimate_tokens

logger = get_logger("reasoning.root_cause_analyzer")

INVESTIGATION_SYSTEM_PROMPT = """You are a Principal SRE performing an evidence-grounded root cause analysis.

EVIDENCE: complete log events in time order, each prefixed by an id like [E00042] (documents: [D00001]). "^ same template xN" lines summarise repeats and list their ids. Nothing else is known about this incident.

RULES
1. Incident-specific claims must cite shown ids in evidence_ids. Never invent ids, services, timestamps or causes.
2. Claim types: OBSERVED = stated by the cited events; INFERRED = reasoned from cited events; LIKELY = best-supported explanation, not proven; CONFIRMED = a cited event explicitly states the cause ("root cause ...", "caused by", "due to"); UNKNOWN = insufficient evidence.
3. Time order is not causation. Never turn co-occurrence, RELATED_TO or PRECEDES into a cause. A metric below 100% is not exhaustion unless an event says so.
4. Keep separate: earliest anomaly, first explicit service failure, propagation to dependents, root cause, recovery. The earliest anomaly or first error is NOT automatically the root cause: prefer the explanation with the strongest corroboration (explicit diagnostic statements, consistent resource exhaustion, recovery after the matching remediation).
5. Vocabulary: anomaly (abnormal signal), degradation (worsening), failure (errors / failing requests), propagation (dependents failing), recovery. Call it a failure only if an ERROR/failure event supports it.
6. If evidence is insufficient, say "Evidence is insufficient to establish this." No generic filler or advice unrelated to the evidence.
7. Timeline: you may relabel skeleton entries (by event_id) only.

Respond ONLY with JSON:
{"executive_summary": "2-4 factual sentences citing ids",
 "earliest_anomaly": {"statement": "...", "evidence_ids": []},
 "root_cause": {"statement": "...", "type": "OBSERVED|INFERRED|LIKELY|CONFIRMED|UNKNOWN", "evidence_ids": []},
 "cause_chain": [{"step": "mechanism step", "type": "...", "evidence_ids": []}],
 "first_service_failure": {"statement": "...", "evidence_ids": []},
 "propagation": [{"text": "...", "type": "...", "evidence_ids": []}],
 "affected_systems": [{"name": "service", "impact": "warning|degraded|failed|unknown", "evidence_ids": []}],
 "recovery": [{"text": "...", "type": "...", "evidence_ids": []}],
 "claims": [{"text": "...", "type": "...", "evidence_ids": []}],
 "timeline": [{"event_id": "E00001", "phase": "PRECURSOR|ANOMALY|DEGRADATION|FAILURE|PROPAGATION|RECOVERY|CONTEXT"}],
 "confidence": 0.0,
 "confidence_explanation": "why, in terms of evidence quality",
 "recommendations": ["action tied to the evidence"],
 "insufficient_evidence": ["unknowns / limitations"]}"""

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
        analysis: Optional[IncidentAnalysis] = None,
    ) -> InvestigationResult:
        if analysis is None:
            analysis = analyze_incident(_groups_of(pack), event_lookup)
        prompt = self.build_prompt(question, pack, timeline, graph_facts)
        estimated = estimate_tokens(INVESTIGATION_SYSTEM_PROMPT) + estimate_tokens(prompt)
        metrics = current_metrics()
        if metrics is not None:
            metrics.incr("prompt_tokens_estimated", estimated)
            metrics.incr("context_budget_tokens", self.settings.MAX_GEMINI_CONTEXT_TOKENS)

        if not pack.items and not pack.document_evidence:
            result = self.deterministic_result(investigation_id, pack, timeline, evidence, "no_evidence", analysis)
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
            result = self.deterministic_result(investigation_id, pack, timeline, evidence, llm.error_kind or "gemini_error",
                                              analysis)
            result.prompt_tokens_estimated = estimated
            return result

        try:
            parsed = self._parse(investigation_id, llm.data, pack, timeline, event_lookup, evidence, analysis)
        except Exception as exc:  # schema-invalid output must degrade, never error
            logger.warning(f"Investigation Gemini output unusable ({type(exc).__name__}); using deterministic fallback")
            parsed = None
        if parsed is None:
            if metrics is not None:
                metrics.record_error("invalid_llm_output")
            result = self.deterministic_result(investigation_id, pack, timeline, evidence, "invalid_llm_output", analysis)
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
        analysis: Optional[IncidentAnalysis] = None,
    ) -> Optional[InvestigationResult]:
        valid = pack.valid_event_ids
        validator = ClaimValidator(valid, event_lookup)

        raw_root = data.get("root_cause")
        if isinstance(raw_root, dict):
            statement = next((raw_root[k].strip() for k in ("statement", "text", "root_cause")
                              if isinstance(raw_root.get(k), str) and raw_root[k].strip()), "")
            if not statement:
                return None  # an object without a statement is a schema error, never str(dict)
            root_claim = validator.validate(statement, raw_root.get("type"), raw_root.get("evidence_ids"))
        elif isinstance(raw_root, str):
            statement = raw_root.strip()
            root_claim = validator.validate(statement, data.get("root_cause_type"), data.get("root_cause_evidence_ids"))
        else:
            return None
        statement = strip_unverified_ids(statement, valid)
        if not statement:
            return None

        chain_claims = validator.validate_items(data.get("cause_chain"), ("step", "text", "statement"))
        claims = validator.validate_items(data.get("claims"))
        milestones: Dict[str, object] = {}
        for key in ("earliest_anomaly", "first_service_failure"):
            raw = data.get(key)
            if isinstance(raw, dict) and coerce_str(raw, ("statement", "text")):
                claim = validator.validate(strip_unverified_ids(coerce_str(raw, ("statement", "text")), valid),
                                           raw.get("type") or "OBSERVED", raw.get("evidence_ids"))
                milestones[key] = claim.model_dump()
                claims.append(Claim(text=f"{key.replace('_', ' ').capitalize()}: {claim.text}", type=claim.type,
                                    evidence_ids=claim.evidence_ids, citations_valid=claim.citations_valid))
        for key in ("propagation", "recovery"):
            items = validator.validate_items(data.get(key))
            if items:
                milestones[key] = [c.model_dump() for c in items]
                claims.extend(Claim(text=f"{key.capitalize()}: {c.text}", type=c.type, evidence_ids=c.evidence_ids,
                                    citations_valid=c.citations_valid) for c in items)
        for claim in chain_claims + claims:
            claim.text = strip_unverified_ids(claim.text, valid)

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
        summary = strip_unverified_ids(coerce_str(data.get("executive_summary") or data.get("incident_summary")), valid)
        recommendations = coerce_str_list(data.get("recommendations"))
        insufficient = coerce_str_list(data.get("insufficient_evidence"))
        label = confidence_label(confidence)
        explanation = data.get("confidence_explanation")
        if isinstance(explanation, str) and explanation.strip():
            explanation = f"{label}: {strip_unverified_ids(explanation.strip(), valid)}"
        else:
            explanation = (f"{label}: as assessed by Gemini ({root_claim.type.value} root cause citing "
                           f"{len(root_claim.evidence_ids)} validated event(s)).")
        if analysis is not None:
            milestones.setdefault("deterministic", analysis.milestones_dict())

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
            confidence_label=label,
            confidence_explanation=explanation,
            milestones=milestones,
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
        self,
        investigation_id: str,
        pack: EvidencePack,
        timeline: Timeline,
        evidence: List[Evidence],
        reason: str,
        analysis: Optional[IncidentAnalysis] = None,
    ) -> InvestigationResult:
        """Evidence-scored RCA without an LLM (see incident_analysis for the scoring rules)."""
        if analysis is None:
            by_id = {i.event.id: i.event for i in pack.items}
            analysis = analyze_incident(_groups_of(pack), by_id.get)
        h = analysis.hypothesis
        anomaly = analysis.earliest_anomaly
        if h is not None:
            root_type, confidence = h.claim_type, h.confidence
            root_ids = (h.finding_ids[:3] + h.exhaustion_ids[:1])
        else:
            root_type = ClaimType.UNKNOWN
            confidence = 0.15 if anomaly is not None else 0.1
            root_ids = [anomaly.id] if anomaly is not None else []
        chain_struct = analysis.chain()
        affected = sorted({i.event.service for i in pack.items if i.event.level in ERROR_LEVELS and i.event.service
                           and not ALERTING_SERVICE_RE.search(i.event.service)})
        insufficient = list(h.limitations) if h is not None else []
        if h is None or root_type != ClaimType.CONFIRMED:
            insufficient.append("No log line explicitly states the causal link; the root cause is assessed from "
                                "corroborating evidence, not proven.")
        root = RootCauseResult(
            root_cause=analysis.root_cause_statement(),
            cause_chain=[step["step"] for step in chain_struct],
            confidence_score=confidence,
            evidence=evidence,
            affected_systems=affected,
            root_cause_type=root_type,
            root_cause_evidence_ids=[i for i in root_ids if i],
            claims=analysis.claims(),
            insufficient_evidence=insufficient,
            source="deterministic",
            confidence_label=confidence_label(confidence),
            confidence_explanation=analysis.confidence_explanation(),
            milestones={"deterministic": analysis.milestones_dict()},
        )
        return InvestigationResult(
            root_cause=root,
            timeline=timeline,
            executive_summary=self._deterministic_summary(root, timeline, pack),
            recommendations=self._evidence_recommendations(analysis)
            + ["Re-run the investigation when the Gemini API is available for a fully reasoned, evidence-cited RCA."],
            degraded=True,
            degradation_reason=reason,
            cause_chain_structured=chain_struct,
        )

    @staticmethod
    def _evidence_recommendations(analysis: IncidentAnalysis) -> List[str]:
        h = analysis.hypothesis
        recs: List[str] = []
        if h is not None and h.kind == "change":
            recs.append(f"Review the change that preceded the incident ({h.mechanism}) before re-deploying it, and add "
                        "canary checks with automatic rollback.")
        elif h is not None:
            where = f" in {h.service}" if h.service else ""
            if "leak" in h.mechanism:
                resource = h.mechanism.replace(" leak", "")
                recs.append(f"Find and fix the {h.mechanism}{where}: ensure every acquired {resource} is released "
                            f"(finally/try-with-resources) and enable leak detection for it.")
            else:
                recs.append(f"Address the {h.mechanism}{where} identified in the logs [{', '.join(h.finding_ids[:2])}].")
            if h.exhaustion_ids:
                recs.append(f"Alert on {h.mechanism.split()[0]}/connection pool utilisation before it reaches exhaustion "
                            f"(exhaustion seen at [{h.exhaustion_ids[0]}]).")
        precursor = analysis.precursor()
        if precursor is not None:
            recs.append(f"Alert on the early signal \"{precursor.message}\" [{precursor.id}] to detect this pattern sooner.")
        if not recs:
            recs.append("Gather additional logs/traces: the current evidence does not name a causal mechanism.")
        return recs

    @staticmethod
    def _deterministic_summary(root: RootCauseResult, timeline: Timeline, pack: EvidencePack) -> str:
        phases = [e.phase.value for e in timeline.events if e.phase and e.phase != TimelinePhase.CONTEXT]
        ordered: List[str] = []
        for p in phases:
            if p not in ordered:
                ordered.append(p)
        # Headline only: the full reasoning is already in root_cause (no duplication).
        headline = root.root_cause.split("]. ", 1)[0] + "]." if "]. " in root.root_cause else root.root_cause.split(". ")[0] + "."
        return (
            f"{headline} "
            f"{pack.stats.get('events', 0)} log events were parsed into {pack.stats.get('groups', 0)} distinct templates; "
            f"{len(pack.items)} complete events were selected as evidence. "
            f"Observed phases: {' -> '.join(ordered) or 'none'}."
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


def _groups_of(pack: EvidencePack) -> List:
    """Distinct event groups present in an evidence pack (fallback when no full analysis is supplied)."""
    seen, groups = set(), []
    for item in pack.items:
        if item.group.id not in seen:
            seen.add(item.group.id)
            groups.append(item.group)
    return groups
