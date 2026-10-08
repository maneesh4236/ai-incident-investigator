from pathlib import Path

from app.services.ingestion.event_processor import TAG_PATTERNS, EventProcessor
from app.services.ingestion.log_loader import LogEventParser

FIX = Path(__file__).parent / "fixtures"


def events_from(text: str):
    return LogEventParser().parse_text(text, "d")


def test_same_template_with_different_numbers_is_one_group():
    events = events_from(
        "2030-01-01 10:00:00 svc WARN latency increased to 700 ms\n"
        "2030-01-01 10:00:01 svc WARN latency increased to 900 ms\n"
        "2030-01-01 10:00:02 other WARN latency increased to 950 ms\n"
        "2030-01-01 10:00:03 svc ERROR latency increased to 990 ms\n"
    )
    groups = EventProcessor().process(events)
    assert len(groups) == 3  # different service / level are never merged
    svc_warn = next(g for g in groups if g.service == "svc" and g.level == "WARN")
    assert svc_warn.count == 2
    assert svc_warn.metric_stats["latency_ms"] == {"first": 700.0, "min": 700.0, "max": 900.0, "last": 900.0}
    assert svc_warn.peak_event_id == events[1].id


def test_dedup_keeps_every_instance_bursts_and_request_ids():
    lines = [f"2030-01-01 10:00:{i:02d} gw ERROR HTTP 500 for request req-{i}" for i in range(3)]
    lines += [f"2030-01-01 10:05:{i:02d} gw ERROR HTTP 500 for request req-{10 + i}" for i in range(2)]
    events = events_from("\n".join(lines))
    (group,) = EventProcessor().process(events)
    assert group.count == 5 and group.event_ids == [e.id for e in events]
    assert len(group.bursts) == 2  # 5-minute gap => a separate occurrence
    assert len(group.distinct_request_ids) == 5
    instances = group.instance_ids(10)
    assert events[0].id in instances and events[3].id in instances and events[-1].id in instances


def test_massive_info_collapses_and_singleton_error_survives():
    lines = [f"2030-01-01 10:{i // 60 % 60:02d}:{i % 60:02d} api INFO request ok id={i} took {i % 50}ms" for i in range(5000)]
    lines.insert(4000, "2030-01-01 11:06:40 db ERROR Deadlock detected on table ORDERS")
    events = events_from("\n".join(lines))
    groups = EventProcessor().process(events)
    assert len(groups) == 2
    error = next(g for g in groups if g.level == "ERROR")
    assert error.count == 1


def test_tags():
    proc = EventProcessor()
    cases = {
        "TIMEOUT": "Connection timed out", "HTTP_5XX": "HTTP 503 Service Unavailable", "POOL": "HikariPool exhausted",
        "LEAK": "Session leak detected", "RECOVERY": "Order processing resumed", "DIAGNOSIS": "Root cause confirmed",
        "DEPLOYMENT": "Deployment started version=v2", "CIRCUIT_BREAKER": "circuit breaker open",
        "RETRY_EXHAUSTED": "retries exhausted after 5 attempts", "INCIDENT_DECLARED": "P1 Incident Triggered",
    }
    for tag, message in cases.items():
        (event,) = events_from(f"2030-01-01 10:00:00 svc WARN {message}")
        assert tag in proc.tags_of(event), (tag, message)
    (err,) = events_from("2030-01-01 10:00:00 svc ERROR recovery failed")
    assert "RECOVERY" not in proc.tags_of(err)
    assert set(TAG_PATTERNS)  # table is documented and non-empty


def test_stack_trace_event_is_tagged_exception():
    events = LogEventParser().parse_file(str(FIX / "bracketed_multiline.txt"), "d")
    EventProcessor().process(events)
    trace = next(e for e in events if e.has_stack_trace)
    assert "EXCEPTION" in trace.tags and "TIMEOUT" in trace.tags


def test_group_prefix_separates_documents():
    a = events_from("2030-01-01 10:00:00 svc ERROR boom")
    b = events_from("2030-01-01 10:00:00 svc ERROR boom")
    ga = EventProcessor().process(a, group_prefix="docA")
    gb = EventProcessor().process(b, group_prefix="docB")
    assert ga[0].id != gb[0].id
