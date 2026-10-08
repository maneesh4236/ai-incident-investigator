import uuid
from pathlib import Path
from unittest.mock import MagicMock

from app.models.schemas import ExtractionResult, GraphEntity, GraphRelationship, RelationType
from app.services.graph.graph_builder import GraphBuilder
from app.services.graph.neo4j_service import Neo4jService
from app.services.ingestion.entity_extractor import EntityExtractor
from app.services.ingestion.event_processor import EventProcessor
from app.services.ingestion.log_loader import LogEventParser

FIX = Path(__file__).parent / "fixtures"


def offline_builder():
    neo = MagicMock()
    neo.is_connected = False
    return GraphBuilder(neo4j_service=neo)


def rel(source, target, rtype, basis):
    return GraphRelationship(id=str(uuid.uuid4()), source=source, target=target, type=rtype,
                             investigation_id="inv", basis=basis)


def test_related_to_and_precedes_never_form_causal_paths():
    gb = offline_builder()
    gb.ingest_batch("inv", ExtractionResult(entities=[], relationships=[
        rel("A", "B", RelationType.RELATED_TO, "co_occurrence"),
        rel("B", "C", RelationType.PRECEDES, "temporal"),
        rel("C", "D", RelationType.CAUSES, "co_occurrence"),  # uncited model output
        rel("X", "D", RelationType.CAUSES, "explicit_text"),
    ]))
    assert gb.upstream_causes("inv", "B") == []
    assert gb.upstream_causes("inv", "C") == []
    assert gb.upstream_causes("inv", "D") == ["X"]


def test_parallel_edges_are_both_kept():
    gb = offline_builder()
    gb.ingest_batch("inv", ExtractionResult(entities=[], relationships=[
        rel("A", "B", RelationType.RELATED_TO, "co_occurrence"),
        rel("A", "B", RelationType.CAUSES, "explicit_text"),
    ]))
    _, edges = gb.export("inv")
    assert {e.type for e in edges} == {RelationType.RELATED_TO, RelationType.CAUSES}
    assert gb.upstream_causes("inv", "B") == ["A"]


def test_event_extraction_semantics():
    events = LogEventParser().parse_file(str(FIX / "pool_leak_cascade.log"), "d")
    groups = EventProcessor().process(events)
    result = EntityExtractor(llm_client=MagicMock(is_configured=False)).extract_from_events(
        "inv", groups, {e.id: e for e in events}
    )
    by_type = {}
    for r in result.relationships:
        by_type.setdefault(r.type, []).append(r)
        assert r.evidence_event_ids, "every relationship carries evidence ids"
    assert not by_type.get(RelationType.CAUSES), "no explicit causal wording -> no CAUSES edges"
    deps = {(r.source, r.target) for r in by_type[RelationType.DEPENDS_ON]}
    assert ("account-service", "ledger-service") in deps
    assert all(r.basis == "temporal" and not r.is_causal for r in by_type[RelationType.PRECEDES])
    names = {e.name for e in result.entities}
    assert {"ledger-service", "SQLTimeoutException", "JDBC"} <= names


def test_explicit_caused_by_creates_causal_edge():
    events = LogEventParser().parse_file(str(FIX / "bracketed_multiline.txt"), "d")
    groups = EventProcessor().process(events)
    result = EntityExtractor(llm_client=MagicMock(is_configured=False)).extract_from_events(
        "inv", groups, {e.id: e for e in events}
    )
    causes = [r for r in result.relationships if r.type == RelationType.CAUSES]
    assert any(r.source == "SocketTimeoutException" and r.target == "SQLTransientConnectionException" for r in causes)
    assert all(r.is_causal for r in causes)


def test_llm_causal_edges_require_valid_citations():
    gb = offline_builder()
    added = gb.add_llm_causal_edges("inv", [
        {"step": "leak", "type": "CONFIRMED", "evidence_ids": ["E1"]},
        {"step": "starvation", "type": "LIKELY", "evidence_ids": ["E2"]},
        {"step": "outage", "type": "INFERRED", "evidence_ids": ["E3"]},
        {"step": "blip", "type": "LIKELY", "evidence_ids": []},
    ])
    assert [(r.source, r.target) for r in added] == [("leak", "starvation")]
    assert added[0].basis == "llm_cited" and added[0].is_causal


def test_neo4j_batch_uses_one_query_per_type():
    svc = Neo4jService.__new__(Neo4jService)
    svc._settings = MagicMock(NEO4J_DATABASE="neo4j")
    tx = MagicMock()
    session = MagicMock()
    session.execute_write.side_effect = lambda work: work(tx)
    svc._driver = MagicMock()
    svc._driver.session.return_value.__enter__.return_value = session
    entities = [GraphEntity(id=str(i), name=f"s{i}", type="SERVICE", investigation_id="inv") for i in range(50)]
    entities += [GraphEntity(id="e", name="err", type="ERROR", investigation_id="inv")]
    rels = [rel(f"s{i}", "err", RelationType.RELATED_TO, "emitted_by") for i in range(50)]
    queries = svc.upsert_batch(entities, rels)
    assert queries == 3 and tx.run.call_count == 3
    assert svc._driver.session.call_count == 1
