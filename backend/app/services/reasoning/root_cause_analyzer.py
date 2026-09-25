"""
Performs root cause analysis by combining:
  - retrieved log/document chunks (vector search)
  - graph causal chains (graph search)
into a single LLM reasoning pass that outputs a structured root cause,
cause chain, and confidence score.

Falls back to a pure graph-centrality heuristic when the LLM is unavailable,
so the pipeline always returns *something* usable.
"""
from __future__ import annotations

from typing import List

from app.core.logging_config import get_logger
from app.models.schemas import (
    Evidence,
    HybridRetrievalResult,
    RootCauseResult,
)
from app.services.graph.graph_builder import GraphBuilder
from app.services.llm.gemini_client import GeminiClient
from app.services.reasoning.evidence_builder import EvidenceBuilder

logger = get_logger("reasoning.root_cause_analyzer")

_SYSTEM_PROMPT = """You are a Principal Site Reliability Engineer performing root cause analysis (RCA).
You are given: (1) relevant log/document excerpts, (2) a knowledge graph context with causal chains.

Respond ONLY with strict JSON in this exact shape:
{
  "root_cause": "short root cause statement",
  "cause_chain": ["Redis Timeout", "Connection Pool Exhaustion", "Payment Failure"],
  "confidence_score": 0.92,
  "affected_systems": ["Payment Service", "Checkout"]
}
Base your answer only on the provided evidence. If evidence is thin, lower the confidence_score."""


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

    # ------------------------------------------------------------------ #
    def _analyze_with_llm(self, retrieval: HybridRetrievalResult, evidence: List[Evidence]) -> RootCauseResult | None:
        chunk_text = "\n---\n".join(e.text for e in evidence[:8])
        graph_paths = "; ".join(" -> ".join(p) for p in retrieval.graph_context.paths) or "none found"
        graph_rels = "; ".join(
            f"{r.source} {r.type.value} {r.target}" for r in retrieval.graph_context.relationships[:15]
        )

        prompt = (
            f"Question: {retrieval.query}\n\n"
            f"Log/document excerpts:\n{chunk_text}\n\n"
            f"Graph causal paths: {graph_paths}\n"
            f"Graph relationships: {graph_rels}\n"
        )
        data = self.llm_client.generate_json(prompt, system_instruction=_SYSTEM_PROMPT)
        if not data.get("root_cause"):
            return None

        return RootCauseResult(
            root_cause=data["root_cause"],
            cause_chain=list(data.get("cause_chain", [])),
            confidence_score=float(data.get("confidence_score", 0.6)),
            evidence=evidence[:8],
            affected_systems=list(data.get("affected_systems", [])),
        )

    def _analyze_heuristic(
        self, investigation_id: str, retrieval: HybridRetrievalResult, evidence: List[Evidence]
    ) -> RootCauseResult:
        paths = retrieval.graph_context.paths
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

        affected = [
            e.name
            for e in retrieval.graph_context.entities
            if e.type.value == "SERVICE"
        ]

        return RootCauseResult(
            root_cause=root_cause,
            cause_chain=cause_chain,
            confidence_score=round(confidence, 2),
            evidence=evidence[:8],
            affected_systems=affected,
        )
