from pathlib import Path

from app.services.ingestion.log_loader import LogEventParser, parse_timestamp

FIX = Path(__file__).parent / "fixtures"


def parse(name):
    return LogEventParser().parse_file(str(FIX / name), "doc")


def test_minimal_format_one_event_per_line():
    events = parse("minimal_incident.log")
    assert len(events) == 7
    assert [e.level for e in events][:3] == ["INFO", "WARN", "ERROR"]
    assert events[2].metrics == {"latency_ms": 3000.0}
    assert events[4].status_code == 502
    assert all(e.ts is not None for e in events)


def test_bracketed_format_service_ids_and_multiline():
    events = parse("bracketed_multiline.txt")
    first = events[0]
    assert first.service == "edge-proxy" and first.level == "INFO"
    assert first.ids == {"request_id": "req-5001"}
    trace = next(e for e in events if "Failed to load cart contents" in e.message)
    assert trace.is_multiline and trace.has_stack_trace
    assert trace.exception_type == "java.sql.SQLTransientConnectionException"
    assert "Caused by: java.net.SocketTimeoutException" in trace.raw
    assert "... 12 more" in trace.raw
    assert trace.raw.count("\n") == trace.continuation_lines == 7
    crit = next(e for e in events if "P1 Incident" in e.message)
    assert crit.level == "CRITICAL" and crit.service == "watchdog-service"


def test_service_before_level_format_and_metrics():
    events = parse("pool_leak_cascade.log")
    pool = next(e for e in events if e.message.startswith("DbPool active=72"))
    assert pool.service == "ledger-service" and pool.level == "WARN"
    assert pool.metrics == {"active": 72.0, "idle": 28.0, "total": 100.0}
    usage = next(e for e in events if e.message == "pool_usage=84%")
    assert usage.metrics == {"pool_pct": 84.0}
    assert next(e for e in events if "txnId=TX90001" in e.message).ids == {"txn_id": "TX90001"}


def test_python_traceback_and_crlf_and_invalid_utf8(tmp_path):
    raw = (
        b"2030-01-01 10:00:00,123 ERROR worker Job failed\r\n"
        b"Traceback (most recent call last):\r\n"
        b'  File "job.py", line 3, in run\r\n'
        b"ValueError: bad \xff input\r\n"
        b"2030-01-01 10:00:01,000 INFO worker next\r\n"
    )
    path = tmp_path / "py.log"
    path.write_bytes(raw)
    events = LogEventParser().parse_file(str(path), "d")
    assert len(events) == 2
    assert events[0].has_stack_trace and events[0].continuation_lines == 3
    assert events[0].exception_type == "ValueError"
    assert "\r" not in events[0].raw
    assert events[0].ts.microsecond == 123000


def test_file_without_timestamps_falls_back_to_lines(tmp_path):
    path = tmp_path / "plain.log"
    path.write_text("ERROR something broke\nINFO all good\nWARN hmm\n", encoding="utf-8")
    events = LogEventParser().parse_file(str(path), "d")
    assert [e.level for e in events] == ["ERROR", "INFO", "WARN"]


def test_level_normalisation_and_ids_are_unique(tmp_path):
    path = tmp_path / "lv.log"
    path.write_text(
        "2030-01-01 10:00:00 WARNING a\n2030-01-01 10:00:01 FATAL b\n2030-01-01 10:00:02 svc ERR c\n", encoding="utf-8"
    )
    events = LogEventParser().parse_file(str(path), "d", id_offset=10)
    assert [e.level for e in events] == ["WARN", "CRITICAL", "ERROR"]
    assert [e.id for e in events] == ["E00011", "E00012", "E00013"]


def test_timestamp_formats():
    assert parse_timestamp("2030-01-01T10:00:00Z").hour == 10
    assert parse_timestamp("2030-01-01 10:00:00+02:00").hour == 8  # normalised to UTC
    assert parse_timestamp("Jan  5 10:00:00").month == 1
    assert parse_timestamp("10:00:00.5").microsecond == 500000
    assert parse_timestamp("not a time") is None


def test_looks_like_log():
    assert LogEventParser.looks_like_log(str(FIX / "bracketed_multiline.txt"))
    assert not LogEventParser.looks_like_log(str(FIX / "gen_logs.py"))
