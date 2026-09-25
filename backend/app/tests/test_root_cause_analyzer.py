from unittest.mock import MagicMock

from app.models.schemas import (
    Chunk,
    HybridRetrievalResult,
    RetrievedChunk,
    RetrievedGraphContext,
)
from app.services.reasoning.root_cause_analyzer import RootCauseAnalyzer


def _retrieval_with_path(path):
    chunk = Chunk(
        id="c1", document_id="d1", investigation_id="inv-1", text="Redis timed out.", chunk_index=0
    )
    return HybridRetrievalResult(
        query="Why did it fail?",
        chunks=[RetrievedChunk(chunk=chunk, score=0.9)],
        graph_context=RetrievedGraphContext(entities=[], relationships=[], paths=[path]),
    )


def test_heuristic_analysis_uses_longest_causal_path():
    graph_builder = MagicMock()
    llm = MagicMock()
    llm.is_configured = False
    analyzer = RootCauseAnalyzer(graph_builder=graph_builder, llm_client=llm)

    retrieval = _retrieval_with_path(["Redis Timeout", "Connection Pool Exhaustion", "Payment Failure"])
    result = analyzer.analyze("inv-1", retrieval)

    assert result.root_cause == "Redis Timeout"
    assert result.cause_chain == ["Redis Timeout", "Connection Pool Exhaustion", "Payment Failure"]
    assert 0 < result.confidence_score <= 1


def test_heuristic_analysis_falls_back_to_centrality_when_no_paths():
    graph_builder = MagicMock()
    graph_builder.centrality_ranked_nodes.return_value = ["Redis", "Payment Service"]
    llm = MagicMock()
    llm.is_configured = False
    analyzer = RootCauseAnalyzer(graph_builder=graph_builder, llm_client=llm)

    retrieval = _retrieval_with_path([])
    retrieval.graph_context.paths = []
    result = analyzer.analyze("inv-1", retrieval)

    assert result.root_cause == "Redis"


def test_llm_analysis_used_when_configured():
    graph_builder = MagicMock()
    llm = MagicMock()
    llm.is_configured = True
    llm.generate_json.return_value = {
        "root_cause": "Redis Timeout",
        "cause_chain": ["Redis Timeout", "Payment Failure"],
        "confidence_score": 0.92,
        "affected_systems": ["Payment Service"],
    }
    analyzer = RootCauseAnalyzer(graph_builder=graph_builder, llm_client=llm)

    retrieval = _retrieval_with_path(["Redis Timeout", "Payment Failure"])
    result = analyzer.analyze("inv-1", retrieval)

    assert result.root_cause == "Redis Timeout"
    assert result.confidence_score == 0.92
    assert result.affected_systems == ["Payment Service"]
