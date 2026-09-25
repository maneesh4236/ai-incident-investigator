"""
Conversational investigation agent.

Given a free-form question about an investigation ("Why did outage happen?",
"What evidence supports this?", "Which service failed first?"), the agent:
  1. Runs hybrid retrieval scoped to the investigation.
  2. Builds an evidence-grounded prompt (chunks + graph context + prior report
     if available).
  3. Asks the LLM to answer, citing which entities/evidence it used.

This keeps the chat experience firmly grounded in retrieved evidence rather
than open-ended chit-chat — it is an investigation tool, not a generic bot.
"""
from __future__ import annotations

from typing import List, Optional

from app.core.logging_config import get_logger
from app.models.schemas import (
    ChatMessage,
    ChatResponse,
    Evidence,
    RCAReport,
)
from app.services.llm.gemini_client import GeminiClient
from app.services.reasoning.evidence_builder import EvidenceBuilder
from app.services.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger("agents.investigation_agent")

_SYSTEM_PROMPT = """You are an AI Incident Investigator answering follow-up questions during a live RCA
investigation. You must ground every answer in the provided evidence and graph context. If the evidence
does not support a confident answer, say so explicitly rather than guessing.

Respond ONLY with strict JSON in this exact shape:
{
  "answer": "concise, specific answer",
  "referenced_entities": ["Redis", "Payment Service"]
}"""


class InvestigationAgent:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        llm_client: GeminiClient | None = None,
        evidence_builder: EvidenceBuilder | None = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.llm_client = llm_client or GeminiClient()
        self.evidence_builder = evidence_builder or EvidenceBuilder()

    def ask(
        self,
        investigation_id: str,
        message: str,
        history: List[ChatMessage],
        report: Optional[RCAReport] = None,
    ) -> ChatResponse:
        retrieval = self.hybrid_retriever.retrieve(investigation_id, message)
        evidence = self.evidence_builder.build(retrieval)

        if self.llm_client.is_configured:
            response = self._answer_with_llm(message, history, retrieval, evidence, report)
            if response:
                return response

        return self._answer_heuristic(message, retrieval, evidence, report)

    # ------------------------------------------------------------------ #
    def _answer_with_llm(self, message, history, retrieval, evidence, report) -> Optional[ChatResponse]:
        history_text = "\n".join(f"{m.role}: {m.content}" for m in history[-6:])
        chunk_text = "\n---\n".join(e.text for e in evidence[:6])
        graph_rels = "; ".join(
            f"{r.source} {r.type.value} {r.target}" for r in retrieval.graph_context.relationships[:10]
        )
        report_context = ""
        if report:
            report_context = (
                f"Existing RCA root cause: {report.root_cause.root_cause} "
                f"(chain: {' -> '.join(report.root_cause.cause_chain)})"
            )

        prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"New question: {message}\n\n"
            f"Evidence:\n{chunk_text}\n\n"
            f"Graph relationships: {graph_rels}\n"
            f"{report_context}"
        )
        data = self.llm_client.generate_json(prompt, system_instruction=_SYSTEM_PROMPT)
        if not data.get("answer"):
            return None

        return ChatResponse(
            answer=data["answer"],
            supporting_evidence=evidence[:5],
            referenced_entities=list(data.get("referenced_entities", [])),
        )

    def _answer_heuristic(self, message, retrieval, evidence: List[Evidence], report) -> ChatResponse:
        if report and any(k in message.lower() for k in ("root cause", "why", "outage", "fail")):
            answer = (
                f"Based on the investigation so far, the likely root cause is "
                f"'{report.root_cause.root_cause}' with {int(report.root_cause.confidence_score * 100)}% "
                f"confidence. Cause chain: {' -> '.join(report.root_cause.cause_chain)}."
            )
        elif evidence:
            answer = (
                "Here is the most relevant evidence I could find for that question: "
                + evidence[0].text[:200]
            )
        else:
            answer = (
                "I couldn't find strong evidence for that in the uploaded documents yet. "
                "Try uploading more logs or rephrasing the question."
            )

        return ChatResponse(
            answer=answer,
            supporting_evidence=evidence[:5],
            referenced_entities=[e.name for e in retrieval.graph_context.entities[:5]],
        )
