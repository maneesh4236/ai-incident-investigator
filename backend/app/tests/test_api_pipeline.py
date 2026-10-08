"""
End-to-end API tests (upload -> investigate -> chat) against the fake Gemini
transport. They assert BOTH correctness and Gemini call boundaries:

  upload = 0 calls, investigate = exactly 1 logical call, chat = 1 per question,
  independent of log size; prompt <= MAX_GEMINI_CONTEXT_TOKENS; evidence <=
  MAX_GEMINI_EVIDENCE_TOKENS; critical evidence survives; events never split;
  Gemini failures degrade deterministically and never leave PROCESSING.
"""
from __future__ import annotations

import inspect
import os
import re
import time
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.repositories.incident_repository import incident_repository
from app.services.reasoning.token_budget import estimate_tokens
from app.tests.conftest import gemini_error
from app.tests.fixtures.gen_logs import generate

FIX = Path(__file__).parent / "fixtures"

POOL_LEAK_KEY_EVIDENCE = [
    "ledger-db query latency increased to 700 ms",
    "DbPool active=72 idle=28 total=100",
    "ledger-service ERROR SQLTimeoutException timeout waiting for connection",
    "Session leak detected in ledger-service",
    "JDBC connection starvation detected",
    "DbPool exhausted",
    "Session leak identified",
    "INCIDENT RESOLVED",
]


def upload(client, path: Path, name: str | None = None):
    with open(path, "rb") as fh:
        return client.post("/api/upload", files={"files": (name or path.name, fh)})


def investigate(client, iid: str):
    return client.post("/api/investigate", json={"investigation_id": iid})


def chat(client, iid: str, message: str):
    return client.post("/api/chat", json={"investigation_id": iid, "message": message})


def evidence_section(prompt: str) -> str:
    start = prompt.index("EVIDENCE")
    end = prompt.index("TIMELINE SKELETON") if "TIMELINE SKELETON" in prompt else len(prompt)
    return prompt[start:end]


def assert_events_whole(prompt: str, source: Path) -> None:
    """Every rendered line is a complete original line (or an explicit marker)."""
    original = {l.rstrip("\r\n") for l in source.read_text(encoding="utf-8").splitlines()}
    stripped = {l.strip() for l in original}
    for line in evidence_section(prompt).splitlines()[1:]:
        if not line.strip() or line.startswith("(Some evidence") or line.startswith("[Not shown]") or line.startswith("- "):
            continue
        m = re.match(r"^\[(E\d{5,})\] (.*)$", line)
        if m:
            assert m.group(2) in original, f"event header not a complete original line: {line!r}"
            continue
        body = line.strip()
        if body.startswith("^ same template") or re.match(r"^\[\d+ (?:more lines|stack lines|continuation lines) omitted\]$", body):
            continue
        assert body in stripped, f"continuation line not a complete original line: {line!r}"


# --------------------------------------------------------------------------- #
# Call policy and budgets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture", ["minimal_incident.log", "bracketed_multiline.txt", "pool_leak_cascade.log"])
def test_call_policy_and_budget_on_fixtures(api_client, gemini, fixture):
    settings = get_settings()
    r = upload(api_client, FIX / fixture)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metrics"]["gemini_logical_calls"] == 0
    assert body["metrics"]["gemini_api_attempts"] == 0
    assert gemini.calls == []
    assert body["documents"][0]["doc_type"] == "LOG"
    iid = body["investigation_id"]

    r = investigate(api_client, iid)
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["metrics"]["gemini_logical_calls"] == 1
    assert len(gemini.calls) == 1
    call = gemini.calls[0]
    assert estimate_tokens(call["prompt"]) + estimate_tokens(call["system"]) <= settings.MAX_GEMINI_CONTEXT_TOKENS
    assert report["metrics"]["evidence_tokens_used"] <= settings.MAX_GEMINI_EVIDENCE_TOKENS
    assert_events_whole(call["prompt"], FIX / fixture)
    assert incident_repository.get_investigation(iid).status.value == "COMPLETED"

    for i, question in enumerate(["Which service failed first?", "What recovered the system?"], start=1):
        r = chat(api_client, iid, question)
        assert r.status_code == 200, r.text
        assert r.json()["metrics"]["gemini_logical_calls"] == 1
        assert len(gemini.calls) == 1 + i


def test_pool_leak_key_evidence_reaches_prompt(api_client, gemini):
    iid = upload(api_client, FIX / "pool_leak_cascade.log").json()["investigation_id"]
    assert investigate(api_client, iid).status_code == 200
    prompt = gemini.calls[0]["prompt"]
    for key in POOL_LEAK_KEY_EVIDENCE:
        assert key in prompt, key


def test_multiline_stack_trace_stays_attached(api_client, gemini):
    iid = upload(api_client, FIX / "bracketed_multiline.txt").json()["investigation_id"]
    investigate(api_client, iid)
    lines = evidence_section(gemini.calls[0]["prompt"]).splitlines()
    idx = next(i for i, l in enumerate(lines) if "Failed to load cart contents" in l)
    following = "\n".join(lines[idx + 1 : idx + 8])
    assert "java.sql.SQLTransientConnectionException" in following
    assert "Caused by: java.net.SocketTimeoutException: Read timed out" in following
    assert "at com.zaxxer.hikari.pool.HikariPool.getConnection" in following


@pytest.mark.parametrize("size", [100_000, 1_000_000] + ([10_000_000] if os.getenv("AETHERLOG_SLOW_TESTS") else []))
def test_gemini_calls_independent_of_size(api_client, gemini, tmp_path, size):
    path = generate(size, str(tmp_path / f"synthetic_{size}.log"))
    r = upload(api_client, Path(path))
    assert r.status_code == 200, r.text
    assert r.json()["metrics"]["gemini_logical_calls"] == 0 and gemini.calls == []
    iid = r.json()["investigation_id"]
    report = investigate(api_client, iid).json()
    assert report["metrics"]["gemini_logical_calls"] == 1 and len(gemini.calls) == 1
    prompt = gemini.calls[0]["prompt"]
    assert estimate_tokens(prompt) + estimate_tokens(gemini.calls[0]["system"]) <= get_settings().MAX_GEMINI_CONTEXT_TOKENS
    for key in POOL_LEAK_KEY_EVIDENCE:
        assert key in prompt, f"critical evidence lost at {size} bytes: {key}"
    # the multiline stack trace near the end of the file is intact
    assert "Transfer persistence failed" in prompt
    assert "Caused by: java.net.SocketTimeoutException: Read timed out" in prompt
    assert_events_whole(prompt, Path(path))
    assert chat(api_client, iid, "What failed first?").json()["metrics"]["gemini_logical_calls"] == 1


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #
def test_invented_event_ids_are_rejected(api_client, gemini):
    iid = upload(api_client, FIX / "pool_leak_cascade.log").json()["investigation_id"]
    events = incident_repository.events_for_investigation(iid)
    leak = next(e for e in events if "Session leak detected" in e.message)
    identified = next(e for e in events if "Session leak identified" in e.message)
    gemini.script.append(
        {
            "root_cause": {"statement": "Session leak in ledger-service starved the DB pool", "type": "CONFIRMED",
                           "evidence_ids": [leak.id, identified.id, "E99999", "bogus"]},
            "cause_chain": [{"step": "Session leak", "type": "CONFIRMED", "evidence_ids": [identified.id]},
                            {"step": "Pool starvation", "type": "OBSERVED", "evidence_ids": ["E88888"]}],
            "claims": [{"text": "Pool was exhausted at 09:05", "type": "OBSERVED", "evidence_ids": ["E00008"]}],
            "timeline": [{"event_id": "E77777", "phase": "FAILURE"}],
            "confidence": "high",
        }
    )
    report = investigate(api_client, iid).json()
    rc = report["root_cause"]
    assert "E99999" not in rc["root_cause_evidence_ids"]
    assert set(rc["root_cause_evidence_ids"]) == {leak.id, identified.id}
    assert rc["root_cause_type"] == "CONFIRMED"  # cites a DIAGNOSIS event ("identified")
    starvation = next(c for c in rc["claims"] if c["text"] == "Pool starvation")
    assert starvation["type"] == "UNKNOWN" and starvation["evidence_ids"] == []
    exhausted = next(c for c in rc["claims"] if "exhausted" in c["text"])
    assert exhausted["type"] == "INFERRED"  # E00008 does not say "exhausted"
    all_ids = {i for e in report["timeline"]["events"] for i in e["event_ids"]}
    assert "E77777" not in all_ids


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "failure, kind, attempts",
    [
        (lambda: gemini_error(429, "RESOURCE_EXHAUSTED"), "rate_limited", 3),
        (lambda: gemini_error(503, "UNAVAILABLE"), "server_error", 3),
        (lambda: gemini_error(400, "INVALID_ARGUMENT"), "client_error", 1),
        (lambda: gemini_error(429, "RESOURCE_EXHAUSTED", {"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}}), "quota_exhausted", 1),
    ],
)
def test_gemini_failures_degrade_deterministically(api_client, gemini, failure, kind, attempts):
    iid = upload(api_client, FIX / "pool_leak_cascade.log").json()["investigation_id"]
    gemini.script.extend([failure() for _ in range(5)])
    r = investigate(api_client, iid)
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["degraded"] is True and report["degradation_reason"] == kind
    assert report["metrics"]["gemini_api_attempts"] == attempts
    assert report["metrics"]["gemini_logical_calls"] == 1
    assert report["root_cause"]["root_cause_type"] == "UNKNOWN"
    assert report["root_cause"]["source"] == "deterministic"
    assert report["timeline"]["events"], "deterministic timeline must still be produced"
    assert incident_repository.get_investigation(iid).status.value == "COMPLETED"
    gemini.script.clear()


def test_gemini_timeout_degrades(api_client, gemini, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "GEMINI_TIMEOUT_SECONDS", 0.2)
    iid = upload(api_client, FIX / "minimal_incident.log").json()["investigation_id"]

    def slow(contents, config):
        time.sleep(1.0)
        return {"root_cause": "late"}

    gemini.script.extend([slow, slow, slow])
    started = time.perf_counter()
    report = investigate(api_client, iid).json()
    assert time.perf_counter() - started < 5
    assert report["degraded"] is True and report["degradation_reason"] == "timeout"
    assert report["metrics"]["gemini_api_attempts"] == 2  # a timeout is retried at most once
    gemini.script.clear()


def test_schema_drift_is_coerced_not_500(api_client, gemini):
    iid = upload(api_client, FIX / "pool_leak_cascade.log").json()["investigation_id"]
    gemini.script.append(
        {
            "root_cause": "Connection pool starvation in ledger-service",
            "cause_chain": [{"event": "a", "ts": "09:05"}, "plain step"],
            "confidence": "high",
            "recommendations": [{"action": "Fix the session leak"}, "Add pool alerts"],
            "affected_systems": [{"name": "ledger-service", "impact": "failed"}],
        }
    )
    r = investigate(api_client, iid)
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["degraded"] is False
    assert report["recommendations"] == ["Fix the session leak", "Add pool alerts"]
    assert report["root_cause"]["root_cause_type"] == "UNKNOWN"  # no type / citations given
    assert report["confidence"] <= 0.3


def test_invalid_json_degrades(api_client, gemini):
    iid = upload(api_client, FIX / "minimal_incident.log").json()["investigation_id"]
    gemini.script.append("this is not json")
    report = investigate(api_client, iid).json()
    assert report["degraded"] is True and report["degradation_reason"] == "invalid_json"


def test_chat_degrades_on_gemini_failure(api_client, gemini):
    iid = upload(api_client, FIX / "pool_leak_cascade.log").json()["investigation_id"]
    gemini.script.extend([gemini_error(503, "UNAVAILABLE")] * 3)
    r = chat(api_client, iid, "Which service failed first?")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["degraded"] is True and body["reasoning_mode"] == "deterministic_fallback"
    assert "earliest abnormal event was [E00005]" in body["answer"]  # WARN latency, an anomaly
    assert "first failure" in body["answer"] and "SQLTimeoutException" in body["answer"]
    assert body["metrics"]["gemini_logical_calls"] == 1 and body["metrics"]["gemini_api_attempts"] == 3


def test_ingestion_failure_never_leaves_processing(api_client, gemini, monkeypatch):
    from app.core.dependencies import get_embedder, get_qdrant_service

    def boom(texts):
        raise RuntimeError("embedding backend down")

    monkeypatch.setattr(get_embedder(), "embed_documents", boom)
    r = upload(api_client, FIX / "pool_leak_cascade.log")
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail["error"] == "ingestion_failed" and detail["stage"] == "embed"
    iid = detail["investigation_id"]
    inv = incident_repository.get_investigation(iid)
    assert inv.status.value == "FAILED" and inv.error
    assert incident_repository.chunks_for_investigation(iid) == []
    assert incident_repository.events_for_investigation(iid) == []
    assert get_qdrant_service().search(iid, [0.1] * get_settings().EMBEDDING_DIM, 5) == []
    assert gemini.calls == []


def test_concurrent_investigation_is_rejected(api_client, gemini):
    iid = upload(api_client, FIX / "minimal_incident.log").json()["investigation_id"]
    assert incident_repository.try_mark_processing(iid)
    assert investigate(api_client, iid).status_code == 409
    incident_repository.ensure_not_processing(iid, incident_repository.get_investigation(iid).status.__class__.PENDING)


def test_empty_and_no_error_logs(api_client, gemini, tmp_path):
    empty = tmp_path / "empty.log"
    empty.write_text("", encoding="utf-8")
    r = upload(api_client, empty)
    assert r.status_code == 200
    assert investigate(api_client, r.json()["investigation_id"]).status_code == 400

    quiet = tmp_path / "quiet.log"
    quiet.write_text("\n".join(f"2030-01-01 10:00:{i:02d} api INFO request ok id={i}" for i in range(30)), encoding="utf-8")
    iid = upload(api_client, quiet).json()["investigation_id"]
    gemini.script.extend([gemini_error(503, "UNAVAILABLE")] * 3)
    report = investigate(api_client, iid).json()
    assert "No WARN/ERROR events" in report["root_cause"]["root_cause"]
    assert report["root_cause"]["root_cause_type"] == "UNKNOWN"


def test_routes_do_not_block_event_loop():
    from app.api import chat as chat_api, graph as graph_api, investigate as inv_api, upload as upload_api

    for fn in (upload_api.upload_documents, inv_api.investigate, chat_api.chat, graph_api.get_graph):
        assert not inspect.iscoroutinefunction(fn), f"{fn.__name__} must be a sync route (threadpool)"
