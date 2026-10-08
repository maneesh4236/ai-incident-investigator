"""
Conversational investigation agent: ONE Gemini call per user question.

For each question the agent
  1. selects complete events under MAX_CHAT_EVIDENCE_TOKENS (question-weighted:
     Qdrant relevance x2, literal id matches are protected),
  2. adds the existing report's root cause (typed) and whole recent history
     messages within MAX_CHAT_HISTORY_TOKENS,
  3. asks Gemini once for a cited, typed answer, and validates the citations.

If Gemini is unavailable the answer is built deterministically from the same
evidence (earliest abnormal event, first ERROR per service, top matches) and
marked degraded - it never raises.
"""
from __future__ import annotations

from typing import List, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import (
    ChatMessage,
    ChatResponse,
    Claim,
    ClaimType,
    Evidence,
    RCAReport,
)
from app.repositories.incident_repository import IncidentRepository, incident_repository
from app.services.ingestion.events import ERROR_LEVELS
from app.services.llm.gemini_client import GeminiClient
from app.services.reasoning.claims import ClaimValidator, coerce_str, coerce_str_list
from app.services.reasoning.evidence_builder import EvidenceBuilder
from app.services.reasoning.evidence_selector import EvidencePack, EvidenceSelector
from app.services.reasoning.investigation_service import group_point_ids
from app.services.reasoning.token_budget import estimate_tokens
from app.services.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger("agents.investigation_agent")

CHAT_SYSTEM_PROMPT = """You are an AI Incident Investigator answering a follow-up question during an RCA investigation.

EVIDENCE: complete log events with ids like [E00042] (documents: [D00001]); "^ same template xN" lines summarise repeats.

RULES
1. Answer strictly from the evidence. Cite ids. Never cite an id that is not shown.
2. Distinguish warning/anomaly, degradation, failure, propagation and recovery. Do not call something a failure unless an
   ERROR/failure event supports it. For "what failed first" questions, separate the earliest abnormal signal from the first failure.
3. Correlation or temporal order is not causation. A metric below 100% is not exhaustion unless an event says so.
4. Classify claims as OBSERVED, INFERRED, LIKELY, CONFIRMED (a cited event explicitly states the cause) or UNKNOWN.
5. If the evidence does not answer the question, say "Evidence is insufficient to establish this."

Respond ONLY with JSON:
{"answer": "concise, specific answer with [E#####] citations",
 "claims": [{"text": "...", "type": "OBSERVED|INFERRED|LIKELY|CONFIRMED|UNKNOWN", "evidence_ids": ["E00001"]}],
 "referenced_entities": ["service-name"],
 "evidence_ids": ["E00001"]}"""

_REPORT_SUMMARY_TOKENS = 300
_SCAFFOLD_TOKENS = 80


class InvestigationAgent:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        llm_client: GeminiClient | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        selector: EvidenceSelector | None = None,
        repo: IncidentRepository | None = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.llm_client = llm_client or GeminiClient()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.selector = selector or EvidenceSelector()
        self.repo = repo or incident_repository
        self.settings = get_settings()

    def ask(
        self,
        investigation_id: str,
        message: str,
        history: List[ChatMessage],
        report: Optional[RCAReport] = None,
    ) -> ChatResponse:
        s = self.settings
        report_context = self._report_context(report)
        history_text = self._history_text(history)
        fixed = (
            estimate_tokens(CHAT_SYSTEM_PROMPT) + estimate_tokens(message) + estimate_tokens(report_context)
            + estimate_tokens(history_text) + _SCAFFOLD_TOKENS
        )
        budget = max(0, min(s.MAX_CHAT_EVIDENCE_TOKENS, s.MAX_GEMINI_CONTEXT_TOKENS - fixed))

        try:
            hits = self.hybrid_retriever.vector_retriever.retrieve(investigation_id, message, top_k=20)
        except Exception as exc:
            logger.warning(f"Chat vector retrieval failed ({exc}); continuing without relevance scores")
            hits = []
        events = self.repo.events_for_investigation(investigation_id)
        groups = self.repo.groups_for_investigation(investigation_id)
        matches = self.repo.find_events(investigation_id, ids=EvidenceSelector.question_ids(message), limit=10)
        pack = self.selector.select(
            events, groups, message, budget_tokens=budget, mode="chat", vector_hits=hits, question_matches=matches
        )
        names = {d.id: d.filename for d in self.repo.documents_for_investigation(investigation_id)}
        pack.append_documents(hits, names, budget)
        evidence = pack.to_evidence(group_point_ids(self.repo, investigation_id), names)

        if budget <= 0:
            return self._answer_deterministic(pack, evidence, "question_too_long")

        prompt = (
            f"CONVERSATION SO FAR:\n{history_text or 'none'}\n\n"
            f"EXISTING RCA: {report_context or 'none yet'}\n\n"
            f"EVIDENCE:\n{pack.rendered or 'none'}\n\n"
            f"QUESTION: {message}"
        )
        result = self.llm_client.call(
            prompt, CHAT_SYSTEM_PROMPT, purpose="chat", json_mode=True,
            max_output_tokens=s.GEMINI_MAX_OUTPUT_TOKENS_CHAT,
        )
        if not result.ok:
            return self._answer_deterministic(pack, evidence, result.error_kind or "gemini_error")

        data = result.data
        answer = coerce_str(data.get("answer"), ("answer", "text"))
        if not answer:
            return self._answer_deterministic(pack, evidence, "invalid_llm_output")

        validator = ClaimValidator(pack.valid_event_ids, lambda eid: self.repo.get_event(investigation_id, eid))
        claims = validator.validate_items(data.get("claims"))
        cited = validator.citations(data.get("evidence_ids"))
        cited_ids = []
        for eid in [i for c in claims for i in c.evidence_ids] + cited:
            if eid not in cited_ids:
                cited_ids.append(eid)
        by_id = {e.event_id: e for e in evidence}
        supporting = [by_id[i] for i in cited_ids if i in by_id]
        supporting += [e for e in sorted(evidence, key=lambda e: -e.relevance) if e not in supporting][: max(0, 5 - len(supporting))]

        return ChatResponse(
            answer=answer,
            supporting_evidence=supporting,
            referenced_entities=coerce_str_list(data.get("referenced_entities")),
            claims=claims,
        )

    # ------------------------------------------------------------------ #
    def _history_text(self, history: List[ChatMessage]) -> str:
        """Newest-first whole messages within MAX_CHAT_HISTORY_TOKENS (never cut a message)."""
        kept: List[str] = []
        used = 0
        for m in reversed(history[-12:]):
            line = f"{m.role}: {m.content}"
            cost = estimate_tokens(line)
            if used + cost > self.settings.MAX_CHAT_HISTORY_TOKENS:
                continue
            kept.insert(0, line)
            used += cost
        return "\n".join(kept)

    @staticmethod
    def _report_context(report: Optional[RCAReport]) -> str:
        if not report:
            return ""
        rc = report.root_cause
        ids = ", ".join(rc.root_cause_evidence_ids[:5])
        text = f"Root cause ({rc.root_cause_type.value}{', cites ' + ids if ids else ''}): {rc.root_cause}"
        if rc.cause_chain:
            text += " | Chain: " + " -> ".join(rc.cause_chain)
        while estimate_tokens(text) > _REPORT_SUMMARY_TOKENS and " -> " in text:
            text = text.rsplit(" -> ", 1)[0]
        if estimate_tokens(text) > _REPORT_SUMMARY_TOKENS:
            text = f"Root cause ({rc.root_cause_type.value}{', cites ' + ids if ids else ''}): see report"
        return text

    def _answer_deterministic(self, pack: EvidencePack, evidence: List[Evidence], reason: str) -> ChatResponse:
        items = pack.items
        if not items and not pack.document_evidence:
            return ChatResponse(
                answer="No evidence has been ingested for this investigation yet, so this cannot be answered.",
                degraded=True,
                degradation_reason=reason,
            )
        claims: List[Claim] = []
        lines = [f"AI reasoning is unavailable ({reason}); here is what the evidence directly shows."]
        first_abnormal = next((i.event for i in items if i.event.level in ("WARN", "ERROR", "CRITICAL")), None)
        first_error = next((i.event for i in items if i.event.level in ERROR_LEVELS), None)
        if first_abnormal is not None:
            lines.append(f"Earliest abnormal event: [{first_abnormal.id}] {first_abnormal.ts_display} "
                         f"{first_abnormal.level} {first_abnormal.service or ''} {first_abnormal.message}")
            claims.append(Claim(text="Earliest abnormal event", type=ClaimType.OBSERVED, evidence_ids=[first_abnormal.id]))
        if first_error is not None:
            lines.append(f"First ERROR: [{first_error.id}] {first_error.ts_display} {first_error.service or ''} "
                         f"{first_error.message}")
            claims.append(Claim(text="First ERROR event", type=ClaimType.OBSERVED, evidence_ids=[first_error.id]))
        top = sorted(items, key=lambda i: -i.score)[:3]
        for item in top:
            if item.event not in (first_abnormal, first_error):
                lines.append(f"Relevant: [{item.event.id}] {item.event.ts_display} {item.event.level or ''} "
                             f"{item.event.service or ''} {item.event.message}")
        lines.append("Causation between these events is not established without further analysis.")
        cited = {i for c in claims for i in c.evidence_ids}
        supporting = [e for e in evidence if e.event_id in cited] + [
            e for e in sorted(evidence, key=lambda e: -e.relevance) if e.event_id not in cited
        ][:3]
        return ChatResponse(
            answer="\n".join(lines),
            supporting_evidence=supporting[:5],
            referenced_entities=sorted({i.event.service for i in top if i.event.service}),
            claims=claims,
            degraded=True,
            degradation_reason=reason,
        )

