"""
Assembles the final RCA report: executive summary, root cause, timeline,
evidence, affected systems, recommendations, and confidence.

No Gemini call is made here: the executive summary and recommendations come
from the single investigation call (or the deterministic fallback), so the
report can never add claims that were not grounded in the evidence prompt.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging_config import get_logger
from app.models.schemas import RCAReport, RootCauseResult, Timeline

logger = get_logger("reasoning.report_generator")


class ReportGenerator:
    def __init__(self, llm_client=None):
        # `llm_client` accepted for backward compatibility; the report makes no LLM call.
        pass

    def generate(
        self,
        investigation_id: str,
        root_cause: RootCauseResult,
        timeline: Timeline,
        *,
        executive_summary: Optional[str] = None,
        recommendations: Optional[List[str]] = None,
        degraded: bool = False,
        degradation_reason: Optional[str] = None,
        omitted_evidence: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> RCAReport:
        summary = executive_summary or self._heuristic_summary(root_cause, timeline)
        recs = recommendations or self._default_recommendations(root_cause)
        affected_systems = root_cause.affected_systems or self._infer_affected(timeline)

        report = RCAReport(
            investigation_id=investigation_id,
            executive_summary=summary,
            root_cause=root_cause,
            timeline=timeline,
            affected_systems=affected_systems,
            recommendations=recs,
            confidence=root_cause.confidence_score,
            degraded=degraded,
            degradation_reason=degradation_reason,
            omitted_evidence=omitted_evidence or [],
            metrics=metrics,
        )
        logger.info(f"Generated RCA report for investigation {investigation_id} (degraded={degraded})")
        return report

    @staticmethod
    def _heuristic_summary(root_cause: RootCauseResult, timeline: Timeline) -> str:
        chain = " -> ".join(root_cause.cause_chain) if root_cause.cause_chain else root_cause.root_cause
        return (
            f"Root cause ({root_cause.root_cause_type.value}, confidence {int(root_cause.confidence_score * 100)}%): "
            f"{root_cause.root_cause} The reconstructed timeline has {len(timeline.events)} entries; chain: {chain}. "
            f"Affected systems: {', '.join(root_cause.affected_systems) or 'not conclusively identified'}."
        )

    @staticmethod
    def _default_recommendations(root_cause: RootCauseResult) -> List[str]:
        recs = [
            "Add automated alerting on the earliest signal in the cause chain to reduce detection time.",
            "Introduce a runbook step to validate this failure mode during future incident triage.",
        ]
        if root_cause.confidence_score < 0.5:
            recs.append("Gather additional logs/traces; current evidence yields low-confidence root cause.")
        return recs

    @staticmethod
    def _infer_affected(timeline: Timeline) -> List[str]:
        services = []
        for e in timeline.events:
            if e.severity == "critical" and e.service and e.service not in services:
                services.append(e.service)
        return services[:8]
