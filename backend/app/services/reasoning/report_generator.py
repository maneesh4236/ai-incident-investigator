"""
Assembles the final RCA report: executive summary, root cause, timeline,
evidence, affected systems, recommendations, and confidence.
"""
from __future__ import annotations

from typing import List

from app.core.logging_config import get_logger
from app.models.schemas import RCAReport, RootCauseResult, Timeline
from app.services.llm.gemini_client import GeminiClient

logger = get_logger("reasoning.report_generator")

_SYSTEM_PROMPT = """You are a Principal SRE writing the executive summary and recommendations
section of an incident RCA report for engineering leadership.
Respond ONLY with strict JSON in this exact shape:
{
  "executive_summary": "2-4 sentence summary in plain English",
  "recommendations": ["Add circuit breaker around Redis calls", "..."]
}
Be specific and actionable. Do not repeat the root cause chain verbatim; add value."""


class ReportGenerator:
    def __init__(self, llm_client: GeminiClient | None = None):
        self.llm_client = llm_client or GeminiClient()

    def generate(
        self,
        investigation_id: str,
        root_cause: RootCauseResult,
        timeline: Timeline,
    ) -> RCAReport:
        summary, recommendations = self._generate_narrative(root_cause, timeline)

        affected_systems = root_cause.affected_systems or self._infer_affected(timeline)

        report = RCAReport(
            investigation_id=investigation_id,
            executive_summary=summary,
            root_cause=root_cause,
            timeline=timeline,
            affected_systems=affected_systems,
            recommendations=recommendations,
            confidence=root_cause.confidence_score,
        )
        logger.info(f"Generated RCA report for investigation {investigation_id}")
        return report

    def _generate_narrative(self, root_cause: RootCauseResult, timeline: Timeline) -> tuple[str, List[str]]:
        if self.llm_client.is_configured:
            chain = " -> ".join(root_cause.cause_chain) or root_cause.root_cause
            events = "; ".join(f"{e.timestamp or '?'}: {e.title}" for e in timeline.events[:10])
            prompt = (
                f"Root cause chain: {chain}\n"
                f"Confidence: {root_cause.confidence_score}\n"
                f"Timeline: {events}\n"
                f"Affected systems: {', '.join(root_cause.affected_systems) or 'unclear'}"
            )
            data = self.llm_client.generate_json(prompt, system_instruction=_SYSTEM_PROMPT)
            summary = data.get("executive_summary")
            recommendations = data.get("recommendations")
            if summary:
                return summary, recommendations or self._default_recommendations(root_cause)

        return self._heuristic_summary(root_cause, timeline), self._default_recommendations(root_cause)

    @staticmethod
    def _heuristic_summary(root_cause: RootCauseResult, timeline: Timeline) -> str:
        chain = " → ".join(root_cause.cause_chain) if root_cause.cause_chain else root_cause.root_cause
        event_count = len(timeline.events)
        return (
            f"Investigation identified '{root_cause.root_cause}' as the likely root cause "
            f"(confidence {int(root_cause.confidence_score * 100)}%). "
            f"The reconstructed timeline spans {event_count} correlated events, with the cause chain "
            f"{chain}. Affected systems: {', '.join(root_cause.affected_systems) or 'not conclusively identified'}."
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
        # Best-effort: surface titles that look like service names (capitalized words).
        candidates = {e.title for e in timeline.events if e.severity in ("warning", "critical")}
        return list(candidates)[:5]
