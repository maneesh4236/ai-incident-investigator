"""
RCA reasoning quality, causal semantics, report lifecycle, model metadata and
Neo4j degradation - all without real Gemini calls.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.models.schemas import ClaimType, GraphRelationship, RelationType, TimelinePhase
from app.repositories.incident_repository import incident_repository as repo
from app.services.ingestion.event_processor import EventProcessor
from app.services.ingestion.log_loader import LogEventParser
from app.services.reasoning.evidence_selector import EvidenceSelector
from app.services.reasoning.incident_analysis import analyze_incident
from app.services.reasoning.timeline_builder import TimelineBuilder
from app.tests.conftest import gemini_error

FIX = Path(__file__).parent / "fixtures"
_ID_RE = re.compile(r"\bE\d{5,}\b")


def analysis_for(text: str):
    events = LogEventParser().parse_text(text, "d")
    groups = EventProcessor().process(events)
    by_id = {e.id: e for e in events}
    return analyze_incident(groups, by_id.get), by_id


def analysis_for_fixture(name: str, graph_facts=()):
    events = LogEventParser().parse_file(str(FIX / name), "d")
    groups = EventProcessor().process(events)
    by_id = {e.id: e for e in events}
    return analyze_incident(groups, by_id.get, graph_facts), by_id, events, groups


def upload(client, name):
    with open(FIX / name, "rb") as fh:
        r = client.post("/api/upload", files={"files": (name, fh)})
    assert r.status_code == 200 and r.json()["metrics"]["gemini_logical_calls"] == 0
    return r.json()["investigation_id"]


# --------------------------------------------------------------------------- #
# I. Session-leak RCA: explicit diagnosis recognised, precursor not promoted
# --------------------------------------------------------------------------- #
def test_session_leak_is_likely_root_cause_and_latency_is_precursor():
    a, by_id, _, _ = analysis_for_fixture("pool_leak_cascade.log")
    h = a.hypothesis
    assert h is not None and h.mechanism == "session leak" and h.service == "ledger-service"
    assert h.claim_type == ClaimType.LIKELY  # no explicit causal statement -> never CONFIRMED
    assert 0.5 <= h.confidence <= 0.75 and a.confidence_label in ("Medium", "High")
    assert {by_id[i].message for i in h.finding_ids} >= {"Session leak detected in ledger-service", "Session leak identified"}
    assert h.exhaustion_ids and h.propagation and h.recovery_ids
    # earliest anomaly (latency WARN) is reported as a precursor, not the root cause
    assert a.earliest_anomaly.message.startswith("ledger-db query latency increased")
    assert a.precursor() is a.earliest_anomaly
    statement = a.root_cause_statement()
    assert statement.startswith("Likely root cause (LIKELY")
    assert "early anomaly/precursor rather than the strongest root-cause evidence" in statement
    assert "not CONFIRMED" in statement
    explanation = a.confidence_explanation()
    assert "explicitly report the session leak" in explanation and "independent sources" in explanation
    assert "only detected after failures began" in explanation  # honest limitation
    for eid in _ID_RE.findall(statement + explanation):  # K/L: every cited id exists
        assert eid in by_id


def test_mechanism_detection_is_generic_not_hardcoded():
    log = "\n".join([
        "2031-05-01 12:00:00 worker-service INFO job started",
        "2031-05-01 12:01:00 worker-service WARN heap usage at 85%",
        "2031-05-01 12:02:00 worker-service WARN GC pause 900ms memory pressure",
        "2031-05-01 12:03:00 worker-service ERROR OutOfMemoryError while processing batch",
        "2031-05-01 12:03:05 api-gateway ERROR HTTP 503 Service Unavailable",
        "2031-05-01 12:04:00 platform-monitor ERROR Memory leak detected in worker-service",
        "2031-05-01 12:05:00 oncall-engineer INFO Memory leak identified in batch cache",
        "2031-05-01 12:06:00 oncall-engineer INFO Restarting worker-service",
        "2031-05-01 12:07:00 worker-service INFO memory usage back to normal",
    ])
    a, _ = analysis_for(log)
    assert a.hypothesis.mechanism == "memory leak" and a.hypothesis.service == "worker-service"
    assert a.hypothesis.claim_type in (ClaimType.LIKELY, ClaimType.INFERRED)


def test_explicit_causal_statement_is_the_only_route_to_confirmed():
    log = "\n".join([
        "2031-05-01 12:00:00 payment-service WARN connection wait time rising",
        "2031-05-01 12:01:00 payment-service ERROR Unable to obtain JDBC Connection",
        "2031-05-01 12:02:00 oracle-db ERROR Session leak detected in payment-service",
        "2031-05-01 12:03:00 sre-engineer INFO Outage caused by session leak in payment-service",
    ])
    a, _ = analysis_for(log)
    assert a.hypothesis.claim_type == ClaimType.CONFIRMED and a.hypothesis.explicit_causal_ids


def test_no_named_mechanism_stays_unknown_with_low_confidence():
    a, _, _, _ = analysis_for_fixture("bracketed_multiline.txt")
    assert a.hypothesis is None and a.confidence_label == "Low"
    assert a.root_cause_statement().startswith("Evidence is insufficient to establish a root cause")


def test_deployment_followed_by_rollback_recovery_is_only_inferred():
    a, _, _, _ = analysis_for_fixture("minimal_incident.log")
    assert a.hypothesis.kind == "change" and a.hypothesis.claim_type == ClaimType.INFERRED
    assert a.confidence_label == "Low"


# --------------------------------------------------------------------------- #
# H. Distinct "first" milestones
# --------------------------------------------------------------------------- #
def test_first_milestones_are_distinct():
    a, by_id, _, _ = analysis_for_fixture("pool_leak_cascade.log")
    assert a.earliest_anomaly.level == "WARN"
    assert a.first_service_failure.service == "ledger-service" and a.first_service_failure.level == "ERROR"
    assert a.first_propagation.service == "account-service"
    assert a.first_unavailability_report.message == "downstream ledger-service unavailable"
    assert a.earliest_anomaly.id != a.first_service_failure.id != a.first_propagation.id


# --------------------------------------------------------------------------- #
# J. Causality: RELATED_TO / PRECEDES / DEPENDS_ON / AFFECTS never become CAUSES
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rtype", [RelationType.RELATED_TO, RelationType.PRECEDES, RelationType.DEPENDS_ON,
                                   RelationType.AFFECTS])
def test_non_causal_relations_never_confirm(rtype):
    fact = GraphRelationship(id=str(uuid.uuid4()), source="session leak", target="ledger-service", type=rtype,
                             investigation_id="i", basis="explicit_text", evidence_event_ids=["E00001"])
    a, _, _, _ = analysis_for_fixture("pool_leak_cascade.log", graph_facts=[fact])
    assert a.hypothesis.claim_type != ClaimType.CONFIRMED and not a.hypothesis.explicit_causal_ids


# --------------------------------------------------------------------------- #
# Timeline semantics & evidence protection
# --------------------------------------------------------------------------- #
def test_timeline_keeps_precursor_and_diagnostic_findings_in_place():
    _, _, events, groups = analysis_for_fixture("pool_leak_cascade.log")
    pack = EvidenceSelector().select(events, groups, "root cause", budget_tokens=4000)
    phases = {e.title: e.phase for e in TimelineBuilder().build_skeleton("i", pack).events}
    assert phases["ledger-db query latency increased to 700 ms"] == TimelinePhase.ANOMALY
    assert phases["Session leak detected in ledger-service"] == TimelinePhase.CONTEXT  # diagnostic, not propagation
    assert phases["downstream ledger-service unavailable"] == TimelinePhase.PROPAGATION


def test_tight_budget_still_protects_diagnosis_leak_and_remediation():
    _, _, events, groups = analysis_for_fixture("pool_leak_cascade.log")
    pack = EvidenceSelector().select(events, groups, "unrelated words about weather", budget_tokens=700)
    shown = {i.event.message for i in pack.items}
    assert {"Session leak detected in ledger-service", "Session leak identified",
            "JDBC connection starvation detected"} <= shown
    assert pack.tokens_used <= 700


# --------------------------------------------------------------------------- #
# Investigation API: deterministic fallback (B/C), Gemini success (A) and schema errors (E/F)
# --------------------------------------------------------------------------- #
def test_investigation_fallback_returns_evidence_scored_rca(api_client, gemini):
    iid = upload(api_client, "pool_leak_cascade.log")
    gemini.script.extend([gemini_error(429, "RESOURCE_EXHAUSTED")] * 3)
    r = api_client.post("/api/investigate", json={"investigation_id": iid})
    assert r.status_code == 200
    report = r.json()
    rc = report["root_cause"]
    assert report["degraded"] is True and rc["source"] == "deterministic"
    assert rc["root_cause_type"] == "LIKELY" and rc["confidence_label"] in ("Medium", "High")
    assert rc["milestones"]["deterministic"]["first_service_failure"]["service"] == "ledger-service"
    assert any("session leak" in rec for rec in report["recommendations"])
    for eid in _ID_RE.findall(rc["root_cause"]) + rc["root_cause_evidence_ids"]:
        assert repo.get_event(iid, eid) is not None
    assert repo.get_investigation(iid).status.value == "COMPLETED"
    assert report["metrics"]["gemini_logical_calls"] == 1


def test_investigation_gemini_success_keeps_structure_and_strips_invented_ids(api_client, gemini):
    iid = upload(api_client, "pool_leak_cascade.log")
    events = repo.events_for_investigation(iid)
    leak = next(e for e in events if e.message == "Session leak identified")
    anomaly = next(e for e in events if e.level == "WARN")
    gemini.script.append({
        "executive_summary": f"A session leak [{leak.id}] exhausted the pool; see also [E99999].",
        "earliest_anomaly": {"statement": "Latency rose first", "evidence_ids": [anomaly.id]},
        "root_cause": {"statement": f"Session leak in ledger-service [{leak.id}] [E99999]", "type": "CONFIRMED",
                       "evidence_ids": [leak.id, "E99999"]},
        "first_service_failure": {"statement": "ledger-service SQL timeouts", "evidence_ids": ["E77777"]},
        "propagation": [{"text": "account-service failed", "type": "OBSERVED", "evidence_ids": []}],
        "confidence": 0.8,
        "confidence_explanation": "explicit diagnosis plus exhaustion",
        "recommendations": ["Fix the leak"],
    })
    report = api_client.post("/api/investigate", json={"investigation_id": iid}).json()
    rc = report["root_cause"]
    assert report["degraded"] is False and rc["source"] == "gemini" and len(gemini.calls) == 1
    assert rc["root_cause_type"] == "CONFIRMED"  # cites a DIAGNOSIS-tagged event
    assert "E99999" not in rc["root_cause"] and "E99999" not in report["executive_summary"]
    assert rc["root_cause_evidence_ids"] == [leak.id]
    assert rc["confidence_explanation"].startswith("High: explicit diagnosis")
    fsf = rc["milestones"]["first_service_failure"]
    assert fsf["type"] == "UNKNOWN" and fsf["evidence_ids"] == []  # invented citation rejected
    assert rc["milestones"]["earliest_anomaly"]["evidence_ids"] == [anomaly.id]


@pytest.mark.parametrize("payload", [
    {"root_cause": ["not", "an", "object"]},
    {"root_cause": {"statement": ""}},
    {"executive_summary": "no root cause at all"},
])
def test_investigation_schema_invalid_falls_back(api_client, gemini, payload):
    iid = upload(api_client, "pool_leak_cascade.log")
    gemini.script.append(payload)
    r = api_client.post("/api/investigate", json={"investigation_id": iid})
    assert r.status_code == 200
    report = r.json()
    assert report["degraded"] is True and report["degradation_reason"] == "invalid_llm_output"
    assert report["root_cause"]["root_cause_type"] == "LIKELY"


# --------------------------------------------------------------------------- #
# M. Report lifecycle
# --------------------------------------------------------------------------- #
def test_report_lifecycle(api_client, gemini):
    assert api_client.get("/api/investigations/does-not-exist").status_code == 404
    iid = upload(api_client, "minimal_incident.log")
    status = api_client.get(f"/api/investigations/{iid}").json()
    assert status["has_report"] is False and status["status"] == "PENDING"
    pending = api_client.get(f"/api/report/{iid}")
    assert pending.status_code == 404 and "No report yet" in pending.json()["detail"]
    assert api_client.post("/api/investigate", json={"investigation_id": iid}).status_code == 200
    status = api_client.get(f"/api/investigations/{iid}").json()
    assert status["has_report"] is True and status["status"] == "COMPLETED"
    report = api_client.get(f"/api/report/{iid}")
    assert report.status_code == 200 and report.json()["investigation_id"] == iid
    assert api_client.get(f"/api/timeline/{iid}").status_code == 200


# --------------------------------------------------------------------------- #
# N. Model metadata
# --------------------------------------------------------------------------- #
def test_meta_reports_configured_model_without_secrets(api_client, monkeypatch):
    from app.core.config import get_settings
    from app.main import model_display_name

    monkeypatch.setattr(get_settings(), "GEMINI_MODEL", "gemini-3.6-flash")
    body = api_client.get("/api/meta").json()
    assert body["llm"]["model"] == "gemini-3.6-flash" and body["llm"]["display_name"] == "Gemini 3.6 Flash"
    assert get_settings().GEMINI_API_KEY not in str(body)
    assert body["graph_store"]["connected"] is False and body["graph_store"]["backend"] == "in-memory"
    assert model_display_name("gemini-2.5-flash-lite") == "Gemini 2.5 Flash Lite"


# --------------------------------------------------------------------------- #
# T. Neo4j unavailable never crashes the pipeline and is diagnosable
# --------------------------------------------------------------------------- #
def test_neo4j_dns_failure_is_graceful_and_credential_free(monkeypatch):
    import app.services.graph.neo4j_service as ns
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "NEO4J_URI", "neo4j+s://user:secretpw@abc123.databases.neo4j.io")
    driver = MagicMock()
    driver.verify_connectivity.side_effect = ValueError("Cannot resolve address abc123.databases.neo4j.io:7687")
    monkeypatch.setattr(ns, "GraphDatabase", MagicMock(driver=MagicMock(return_value=driver)))
    service = ns.Neo4jService()
    assert service.is_connected is False
    assert "DNS resolution failed" in service.unavailable_reason
    assert ns._safe_target(settings.NEO4J_URI) == "neo4j+s://abc123.databases.neo4j.io:7687"
    assert "secretpw" not in service.unavailable_reason
    assert service.upsert_batch([], []) == 0 and service.get_graph("x") == ([], [])
