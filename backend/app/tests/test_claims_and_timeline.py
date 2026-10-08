from pathlib import Path

from app.models.schemas import ClaimType, TimelinePhase
from app.services.ingestion.event_processor import EventProcessor
from app.services.ingestion.log_loader import LogEventParser
from app.services.reasoning.claims import ClaimValidator, coerce_confidence, normalize_id
from app.services.reasoning.evidence_selector import EvidenceSelector
from app.services.reasoning.timeline_builder import TimelineBuilder

FIX = Path(__file__).parent / "fixtures"


def load(name):
    events = LogEventParser().parse_file(str(FIX / name), "d")
    groups = EventProcessor().process(events)
    return events, groups, {e.id: e for e in events}


def test_claim_rules():
    events, _, by_id = load("pool_leak_cascade.log")
    leak_detected = next(e for e in events if "Session leak detected" in e.message)
    leak_identified = next(e for e in events if "Session leak identified" in e.message)
    pool_84 = next(e for e in events if e.message == "pool_usage=84%")
    exhausted = next(e for e in events if e.message == "DbPool exhausted")
    v = ClaimValidator(set(by_id), by_id.get)

    assert v.validate("x", "OBSERVED", ["E99999"]).type == ClaimType.UNKNOWN
    assert "E99999" in v.rejected_ids
    assert v.validate("x", "LIKELY", []).type == ClaimType.INFERRED
    assert v.validate("leak", "CONFIRMED", [leak_detected.id]).type == ClaimType.LIKELY  # "detected" is not a diagnosis
    assert v.validate("leak", "CONFIRMED", [leak_identified.id]).type == ClaimType.CONFIRMED
    assert v.validate("pool exhausted at 84%", "OBSERVED", [pool_84.id]).type == ClaimType.INFERRED
    assert v.validate("pool exhausted", "OBSERVED", [exhausted.id]).type == ClaimType.OBSERVED
    assert v.validate("x", "nonsense", [exhausted.id]).type == ClaimType.UNKNOWN
    assert normalize_id("[E12]") == "E00012" and normalize_id("foo") is None


def test_confidence_coercion():
    assert coerce_confidence("high") == 0.8
    assert coerce_confidence("85%") == 0.85
    assert coerce_confidence(92) == 0.92
    assert coerce_confidence("garbage", 0.4) == 0.4
    assert coerce_confidence(-3) == 0.0


def _timeline(name):
    events, groups, _ = load(name)
    pack = EvidenceSelector().select(events, groups, "root cause", budget_tokens=4000)
    return TimelineBuilder().build_skeleton("inv", pack), events


def test_pool_leak_phases_follow_rules():
    timeline, _ = _timeline("pool_leak_cascade.log")
    phase_of = {e.title: e.phase for e in timeline.events}
    assert phase_of["ledger-db query latency increased to 700 ms"] == TimelinePhase.ANOMALY
    assert phase_of["DbPool active=72 idle=28 total=100"] == TimelinePhase.DEGRADATION
    assert phase_of["SQLTimeoutException timeout waiting for connection"] == TimelinePhase.FAILURE
    assert phase_of["downstream ledger-service unavailable"] == TimelinePhase.PROPAGATION
    assert phase_of["HTTP 503 Service Unavailable"] == TimelinePhase.PROPAGATION
    assert phase_of["P1 INCIDENT DECLARED"] == TimelinePhase.CONTEXT
    assert phase_of["Account processing resumed"] == TimelinePhase.RECOVERY
    # chronological order and the first WARN is an anomaly, not a failure
    stamps = [e.timestamp for e in timeline.events]
    assert stamps == sorted(stamps)
    assert timeline.events[0].phase in (TimelinePhase.CONTEXT, TimelinePhase.ANOMALY)


def test_minimal_incident_precursor_and_recovery():
    timeline, _ = _timeline("minimal_incident.log")
    phases = [e.phase for e in timeline.events]
    assert phases[0] == TimelinePhase.PRECURSOR  # deployment before the first anomaly
    assert TimelinePhase.FAILURE in phases and phases[-1] == TimelinePhase.RECOVERY


def test_phase_overrides_only_apply_to_existing_entries():
    timeline, events = _timeline("pool_leak_cascade.log")
    first = timeline.events[0]
    applied = TimelineBuilder.apply_phase_overrides(
        timeline, {first.event_ids[0]: TimelinePhase.PRECURSOR, "E99999": TimelinePhase.FAILURE}
    )
    assert applied == 1 and timeline.events[0].phase == TimelinePhase.PRECURSOR
    assert all("E99999" not in e.event_ids for e in timeline.events)


def test_repeated_groups_are_one_timeline_entry_with_instance_ids():
    timeline, _ = _timeline("pool_leak_cascade.log")
    sql = [e for e in timeline.events if e.title.startswith("SQLTimeoutException")]
    assert len(sql) == 1 and sql[0].occurrences == 3 and len(sql[0].event_ids) >= 2


def test_timeline_cap_keeps_diagnosis_and_declaration(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "TIMELINE_MAX_EVENTS", 10)
    timeline, _ = _timeline("pool_leak_cascade.log")
    titles = [e.title for e in timeline.events]
    assert len(timeline.events) == 10
    assert "Session leak identified" in titles
    assert "P1 INCIDENT DECLARED" in titles
    assert any(e.phase == TimelinePhase.RECOVERY for e in timeline.events)
    assert any(e.phase == TimelinePhase.ANOMALY for e in timeline.events)
