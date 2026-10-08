"""
Conversational investigation agent: ONE Gemini call per user question.

For each question the agent
  1. selects complete events under MAX_CHAT_EVIDENCE_TOKENS (question-weighted:
     Qdrant relevance x2, literal id matches are protected),
  2. adds the existing report's root cause (typed) and whole recent history
     messages within MAX_CHAT_HISTORY_TOKENS,
  3. asks Gemini once (bounded attempts, CHAT_GEMINI_DEADLINE_SECONDS wall
     clock) for a cited, typed answer, and validates the citations.

Gemini is the primary reasoning engine (reasoning_mode="gemini"). If the call
fails - 503/high demand, timeout, transport error, invalid output, not
configured - the question is answered deterministically from the SAME
evidence by `DeterministicChatAnswerer` (reasoning_mode="deterministic_fallback",
degraded=True). No extra Gemini call is made; it never raises for Gemini errors.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import List, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import ChatMessage, ChatResponse, Evidence, RCAReport
from app.repositories.incident_repository import IncidentRepository, incident_repository
from app.services.agents.chat_fallback import DeterministicChatAnswerer
from app.services.llm.gemini_client import GeminiClient, outcome_category
from app.services.reasoning.claims import ClaimValidator, coerce_str_list, normalize_id
from app.services.reasoning.evidence_builder import EvidenceBuilder
from app.services.reasoning.evidence_selector import EvidencePack, EvidenceSelector
from app.services.reasoning.investigation_service import group_point_ids
from app.services.reasoning.token_budget import estimate_tokens
from app.services.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger("agents.investigation_agent")

CHAT_SYSTEM_PROMPT = """You are an AI Incident Investigator answering a follow-up question during an RCA investigation.

EVIDENCE: complete log events with ids like [E00042] (documents: [D00001]); "^ same template xN" lines summarise repeats.

RULES
1. Answer the user's exact question first (1-3 sentences), then give the supporting evidence. No generic filler.
2. Use only the evidence; cite ids for important claims. Never cite an id that is not shown or invent services/times.
3. Keep separate: earliest anomaly, first service failure, propagation to dependents, root cause, recovery. Do not call
   something a failure unless an ERROR/failure event supports it. The first anomaly or first error is not automatically
   the root cause; prefer the explanation with the strongest corroboration (explicit diagnostic statements, consistent
   resource exhaustion, recovery after the matching remediation).
4. Time order is not causation; never treat co-occurrence, RELATED_TO or PRECEDES as a cause. A metric below 100% is not
   exhaustion unless an event says so.
5. Classify claims as OBSERVED, INFERRED, LIKELY, CONFIRMED (a cited event explicitly states the cause) or UNKNOWN.
6. If the evidence does not answer the question, say "Evidence is insufficient to establish this."

Respond ONLY with JSON:
{"answer": "concise, specific answer with [E#####] citations",
 "claims": [{"text": "...", "type": "OBSERVED|INFERRED|LIKELY|CONFIRMED|UNKNOWN", "evidence_ids": ["E00001"]}],
 "referenced_entities": ["service-name"],
 "evidence_ids": ["E00001"]}"""

_REPORT_SUMMARY_TOKENS = 300
_SCAFFOLD_TOKENS = 80
_ID_IN_TEXT_RE = re.compile(r"\[?\b([ED]\d{5,})\b\]?")


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
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
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
            logger.warning(f"Chat vector retrieval failed ({type(exc).__name__}); continuing without relevance scores")
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

        def finish(response: ChatResponse, attempts: int) -> ChatResponse:
            logger.info(
                f"chat request_id={request_id} investigation={investigation_id} reasoning_mode={response.reasoning_mode} "
                f"attempts={attempts} outcome={outcome_category(response.degradation_reason)} "
                f"fallback_reason={response.degradation_reason or 'none'} degraded={response.degraded} "
                f"evidence_events={len(pack.items)} evidence_tokens={pack.tokens_used} "
                f"cited_ids={len(response.evidence_ids)} latency_ms={(time.perf_counter() - started) * 1000:.0f}"
            )
            return response

        if budget <= 0:
            return finish(self._answer_deterministic(investigation_id, message, pack, evidence, report,
                                                     "question_too_long"), 0)

        prompt = (
            f"CONVERSATION SO FAR:\n{history_text or 'none'}\n\n"
            f"EXISTING RCA: {report_context or 'none yet'}\n\n"
            f"EVIDENCE:\n{pack.rendered or 'none'}\n\n"
            f"QUESTION: {message}"
        )
        result = self.llm_client.call(
            prompt, CHAT_SYSTEM_PROMPT, purpose="chat", json_mode=True,
            max_output_tokens=s.GEMINI_MAX_OUTPUT_TOKENS_CHAT,
            deadline_seconds=s.CHAT_GEMINI_DEADLINE_SECONDS,
        )
        if not result.ok:
            return finish(self._answer_deterministic(investigation_id, message, pack, evidence, report,
                                                     result.error_kind or "gemini_error"), result.attempts)

        try:
            response = self._gemini_response(investigation_id, result.data, pack, evidence)
        except Exception as exc:  # malformed-but-parseable output must never surface as an error
            logger.warning(f"chat request_id={request_id} could not use Gemini output ({type(exc).__name__})")
            response = None
        if response is None:
            return finish(self._answer_deterministic(investigation_id, message, pack, evidence, report,
                                                     "invalid_llm_output"), result.attempts)
        return finish(response, result.attempts)

    # ------------------------------------------------------------------ #
    def _gemini_response(
        self, investigation_id: str, data: dict, pack: EvidencePack, evidence: List[Evidence]
    ) -> Optional[ChatResponse]:
        raw_answer = data.get("answer")
        if isinstance(raw_answer, dict):  # tolerate {"answer": {"text": "..."}}, nothing else
            raw_answer = next((raw_answer[k] for k in ("text", "answer") if isinstance(raw_answer.get(k), str)), None)
        answer = raw_answer.strip() if isinstance(raw_answer, str) else ""
        if not answer:
            return None
        valid = pack.valid_event_ids
        # Ids in the answer text that were not in the evidence are removed, never shown as citations.
        answer = _ID_IN_TEXT_RE.sub(
            lambda m: m.group(0) if normalize_id(m.group(1)) in valid else "[unverified id removed]", answer
        )

        validator = ClaimValidator(valid, lambda eid: self.repo.get_event(investigation_id, eid))
        claims = validator.validate_items(data.get("claims"))
        cited = validator.citations(data.get("evidence_ids"))
        cited_ids: List[str] = []
        for eid in [i for c in claims for i in c.evidence_ids] + cited + [
            normalize_id(m.group(1)) for m in _ID_IN_TEXT_RE.finditer(answer)
        ]:
            if eid and eid in valid and eid not in cited_ids:
                cited_ids.append(eid)
        by_id = {e.event_id: e for e in evidence}
        supporting = [by_id[i] for i in cited_ids if i in by_id]
        supporting += [e for e in sorted(evidence, key=lambda e: -e.relevance) if e not in supporting][: max(0, 5 - len(supporting))]

        return ChatResponse(
            answer=answer,
            supporting_evidence=supporting,
            referenced_entities=coerce_str_list(data.get("referenced_entities")),
            claims=claims,
            evidence_ids=cited_ids,
            reasoning_mode="gemini",
            degraded=False,
        )

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

    def _answer_deterministic(
        self,
        investigation_id: str,
        question: str,
        pack: EvidencePack,
        evidence: List[Evidence],
        report: Optional[RCAReport],
        reason: str,
    ) -> ChatResponse:
        """Question-aware answer from the same evidence, with no LLM call."""
        try:
            graph_facts = self.hybrid_retriever.graph_retriever.graph_builder.explicit_relationships(investigation_id)
        except Exception:
            graph_facts = []
        answerer = DeterministicChatAnswerer(
            question,
            pack,
            self.repo.groups_for_investigation(investigation_id),
            lambda eid: self.repo.get_event(investigation_id, eid),
            report=report,
            graph_facts=graph_facts,
        )
        result = answerer.answer(reason)
        cited = set(result.evidence_ids)
        supporting = [e for e in evidence if e.event_id in cited] + [
            e for e in sorted(evidence, key=lambda e: -e.relevance) if e.event_id not in cited
        ]
        services = []
        for eid in result.evidence_ids:
            event = self.repo.get_event(investigation_id, eid)
            if event is not None and event.service and event.service not in services:
                services.append(event.service)
        return ChatResponse(
            answer=result.answer,
            supporting_evidence=supporting[:8],
            referenced_entities=services[:8],
            claims=result.claims,
            evidence_ids=result.evidence_ids,
            reasoning_mode="deterministic_fallback",
            degraded=True,
            degradation_reason=reason,
        )
