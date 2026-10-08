"""
Orchestrates one investigation run:

  vector retrieval (relevance signal only)
  -> evidence selection under the hard evidence budget
  -> deterministic timeline skeleton
  -> evidence-backed graph facts
  -> ONE Gemini reasoning call (or deterministic fallback)
  -> validated CAUSES edges back into the graph
  -> report assembly

The caller (API route) owns the metrics scope that limits this to one
logical Gemini call.
"""
from __future__ import annotations

from typing import Dict, List

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.metrics import current_metrics
from app.models.schemas import HybridRetrievalResult, RCAReport, RetrievedGraphContext
from app.repositories.incident_repository import IncidentRepository
from app.services.graph.graph_builder import GraphBuilder
from app.services.reasoning.evidence_selector import EvidenceSelector
from app.services.reasoning.report_generator import ReportGenerator
from app.services.reasoning.root_cause_analyzer import RootCauseAnalyzer
from app.services.reasoning.timeline_builder import TimelineBuilder
from app.services.vector.vector_retriever import VectorRetriever

logger = get_logger("reasoning.investigation_service")

_RELEVANCE_TOP_K = 20


def group_point_ids(repo: IncidentRepository, investigation_id: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for chunk in repo.chunks_for_investigation(investigation_id):
        gid = (chunk.metadata or {}).get("group_id")
        if gid:
            mapping[gid] = chunk.id
    return mapping


def graph_fact_lines(graph_builder: GraphBuilder, investigation_id: str) -> List[str]:
    lines = []
    for rel in graph_builder.explicit_relationships(investigation_id):
        ids = ", ".join(rel.evidence_event_ids[:3])
        lines.append(f"- {rel.source} {rel.type.value} {rel.target}" + (f" [{ids}]" if ids else ""))
    return lines


class InvestigationService:
    def __init__(
        self,
        repo: IncidentRepository,
        vector_retriever: VectorRetriever,
        graph_builder: GraphBuilder,
        selector: EvidenceSelector,
        timeline_builder: TimelineBuilder,
        analyzer: RootCauseAnalyzer,
        report_generator: ReportGenerator,
    ):
        self.repo = repo
        self.vector_retriever = vector_retriever
        self.graph_builder = graph_builder
        self.selector = selector
        self.timeline_builder = timeline_builder
        self.analyzer = analyzer
        self.report_generator = report_generator
        self.settings = get_settings()

    def run(self, investigation_id: str, question: str) -> RCAReport:
        metrics = current_metrics()
        stage = metrics.stage if metrics is not None else _null_stage

        events = self.repo.events_for_investigation(investigation_id)
        groups = self.repo.groups_for_investigation(investigation_id)
        document_names = {d.id: d.filename for d in self.repo.documents_for_investigation(investigation_id)}
        point_ids = group_point_ids(self.repo, investigation_id)
        if metrics is not None:
            metrics.parsed_events, metrics.groups = len(events), len(groups)

        with stage("retrieval"):
            try:
                hits = self.vector_retriever.retrieve(investigation_id, question, top_k=_RELEVANCE_TOP_K)
            except Exception as exc:  # retrieval is only a relevance signal; never fatal
                logger.warning(f"Vector retrieval failed ({exc}); continuing without relevance scores")
                hits = []

        budget = self.analyzer.evidence_budget_for(question)
        with stage("evidence_selection"):
            pack = self.selector.select(events, groups, question, budget_tokens=budget, vector_hits=hits)
            pack.append_documents(hits, document_names, budget)
        evidence = pack.to_evidence(point_ids, document_names)

        with stage("timeline"):
            timeline = self.timeline_builder.build_skeleton(investigation_id, pack, point_ids)
            if not timeline.events and hits:  # document-only investigations
                timeline = self.timeline_builder.build(
                    investigation_id,
                    HybridRetrievalResult(
                        query=question, chunks=list(hits), graph_context=RetrievedGraphContext(entities=[], relationships=[])
                    ),
                )

        facts = graph_fact_lines(self.graph_builder, investigation_id)
        with stage("reasoning"):
            result = self.analyzer.reason(
                investigation_id,
                question,
                pack,
                timeline,
                facts,
                lambda eid: self.repo.get_event(investigation_id, eid),
                evidence,
            )

        if not result.degraded and result.cause_chain_structured:
            self.graph_builder.add_llm_causal_edges(investigation_id, result.cause_chain_structured)

        report = self.report_generator.generate(
            investigation_id,
            result.root_cause,
            result.timeline,
            executive_summary=result.executive_summary,
            recommendations=result.recommendations,
            degraded=result.degraded,
            degradation_reason=result.degradation_reason,
            omitted_evidence=pack.omitted,
        )
        if result.rejected_ids or result.downgrades:
            logger.info(
                f"Claim validation: {len(result.rejected_ids)} rejected ids, {len(result.downgrades)} downgrades"
            )
        return report


class _NullStage:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _null_stage(_name: str) -> _NullStage:
    return _NullStage()
