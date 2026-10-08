"""
Chat reliability: Gemini stays the primary reasoning engine (one logical call
per question); when it fails, the question is still answered deterministically
from the same evidence - fast, grounded, with valid ids and no causal overreach.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.models.schemas import ClaimType, GraphRelationship, RelationType
from app.repositories.incident_repository import incident_repository as repo
from app.services.agents.chat_fallback import DeterministicChatAnswerer, classify_intent
from app.services.ingestion.event_processor import EventProcessor
from app.services.ingestion.log_loader import LogEventParser
from app.services.reasoning.evidence_selector import EvidenceSelector
from app.services.reasoning.token_budget import estimate_tokens
from app.tests.conftest import gemini_error

FIX = Path(__file__).parent / "fixtures"
_ID_RE = re.compile(r"\b[ED]\d{5,}\b")


def upload(client, name):
    with open(FIX / name, "rb") as fh:
        r = client.post("/api/upload", files={"files": (name, fh)})
    assert r.status_code == 200, r.text
    assert r.json()["metrics"]["gemini_logical_calls"] == 0  # upload stays at 0 Gemini calls
    return r.json()["investigation_id"]


def chat(client, iid, message):
    return client.post("/api/chat", json={"investigation_id": iid, "message": message})


def fail_503(gemini, n=3):
    gemini.script.extend([gemini_error(503, "UNAVAILABLE") for _ in range(n)])


def assert_ids_valid(iid, body):
    ids = set(_ID_RE.findall(body["answer"])) | set(body["evidence_ids"])
    for claim in body["claims"]:
        ids |= set(claim["evidence_ids"])
    for eid in ids:
        assert repo.get_event(iid, eid) is not None, f"invented/unknown evidence id {eid}"
    return ids


# --------------------------------------------------------------------------- #
# 1. Gemini success
# --------------------------------------------------------------------------- #
def test_gemini_success_returns_gemini_answer(api_client, gemini):
    iid = upload(api_client, "pool_leak_cascade.log")
    leak = next(e for e in repo.events_for_investigation(iid) if "Session leak identified" in e.message)
    gemini.script.append({
        "answer": f"The logs identify a session leak [{leak.id}]; an id that was never shown is [E99999].",
        "claims": [{"text": "Session leak identified", "type": "OBSERVED", "evidence_ids": [leak.id, "E88888"]}],
        "referenced_entities": ["ledger-service"],
        "evidence_ids": [leak.id, "E77777"],
    })
    r = chat(api_client, iid, "Why did the outage happen?")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reasoning_mode"] == "gemini" and body["degraded"] is False
    assert f"[{leak.id}]" in body["answer"]
    assert "E99999" not in body["answer"] and "[unverified id removed]" in body["answer"]  # 8. no invented ids
    assert body["evidence_ids"] == [leak.id]
    assert body["claims"][0]["evidence_ids"] == [leak.id]
    assert body["metrics"]["gemini_logical_calls"] == 1 and len(gemini.calls) == 1  # 11. one logical call
    assert_ids_valid(iid, body)


# --------------------------------------------------------------------------- #
# 2-4. Failures -> deterministic fallback that answers the question
# --------------------------------------------------------------------------- #
def test_gemini_503_falls_back_with_useful_answer(api_client, gemini):
    iid = upload(api_client, "pool_leak_cascade.log")
    fail_503(gemini)
    r = chat(api_client, iid, "Why did the outage happen?")
    assert r.status_code == 200
    body = r.json()
    assert body["reasoning_mode"] == "deterministic_fallback" and body["degraded"] is True
    assert body["degradation_reason"] == "server_error"
    assert body["metrics"]["gemini_logical_calls"] == 1 and body["metrics"]["gemini_api_attempts"] == 3
    answer = body["answer"]
    assert "AI reasoning is unavailable" not in answer
    assert "earliest abnormal event was [E00005]" in answer  # first WARN: ledger-db latency
    assert "first failure [E" in answer and "SQLTimeoutException" in answer
    assert "Session leak identified" in answer  # diagnosis statement written in the logs
    assert len(assert_ids_valid(iid, body)) >= 4
    # no internal error details leak into the answer
    assert "UNAVAILABLE" not in answer and "Traceback" not in answer and "503" not in answer.split("(Note:")[1]


def test_gemini_timeout_falls_back_quickly(api_client, gemini, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "GEMINI_TIMEOUT_SECONDS", 0.3)
    iid = upload(api_client, "pool_leak_cascade.log")

    def slow(contents, config):
        time.sleep(1.5)
        return {"answer": "too late"}

    gemini.script.extend([slow, slow, slow])
    started = time.perf_counter()
    body = chat(api_client, iid, "What services were affected?").json()
    assert time.perf_counter() - started < 3
    assert body["reasoning_mode"] == "deterministic_fallback" and body["degradation_reason"] == "timeout"
    assert "ledger-service" in body["answer"]
    assert body["metrics"]["gemini_logical_calls"] == 1


def test_chat_deadline_caps_a_hanging_gemini(api_client, gemini, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "GEMINI_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(settings, "CHAT_GEMINI_DEADLINE_SECONDS", 0.5)
    iid = upload(api_client, "minimal_incident.log")

    def hang(contents, config):
        time.sleep(2.0)
        return {"answer": "too late"}

    gemini.script.extend([hang, hang])
    started = time.perf_counter()
    body = chat(api_client, iid, "What happened?").json()
    assert time.perf_counter() - started < 1.8  # deadline, not the 5 s timeout x retries
    assert body["reasoning_mode"] == "deterministic_fallback" and body["metrics"]["gemini_api_attempts"] == 1


@pytest.mark.parametrize("payload, reason", [
    ("definitely not json", "invalid_json"),
    ({"unexpected": "shape"}, "invalid_llm_output"),
    ({"answer": {"nested": True}, "claims": "garbage"}, "invalid_llm_output"),
])
def test_invalid_gemini_output_falls_back(api_client, gemini, payload, reason):
    iid = upload(api_client, "pool_leak_cascade.log")
    gemini.script.append(payload)
    r = chat(api_client, iid, "What was the impact?")
    assert r.status_code == 200
    body = r.json()
    assert body["reasoning_mode"] == "deterministic_fallback" and body["degradation_reason"] == reason
    assert "web-gateway" in body["answer"]
    assert body["metrics"]["gemini_logical_calls"] == 1


# --------------------------------------------------------------------------- #
# 5-7, 16. Question-aware answers
# --------------------------------------------------------------------------- #
def test_affected_services_answer_uses_only_real_services(api_client, gemini):
    iid = upload(api_client, "pool_leak_cascade.log")
    fail_503(gemini)
    body = chat(api_client, iid, "What services were affected?").json()
    real_services = {e.service for e in repo.events_for_investigation(iid) if e.service}
    named = re.findall(r"^\s+- ([\w.\-]+):", body["answer"], re.M)
    assert {"ledger-service", "account-service", "web-gateway", "ledger-db"} <= set(named)
    assert set(named) <= real_services
    assert "pager-alerts" not in named  # alerting source is not an affected service
    assert_ids_valid(iid, body)


def test_impact_answer(api_client, gemini):
    iid = upload(api_client, "pool_leak_cascade.log")
    fail_503(gemini)
    answer = chat(api_client, iid, "What was the impact?").json()["answer"]
    assert "User-facing 5xx responses: web-gateway x1" in answer
    assert "Failure window: from the first failure [E" in answer
    assert "account-service depends on ledger-service" in answer


def test_answers_differ_by_question_and_are_useful(api_client, gemini):
    iid = upload(api_client, "pool_leak_cascade.log")
    answers = {}
    for q in ["Why did the outage happen?", "Which services were impacted?", "When did it start?",
              "How did it recover?", "What was the timeline?", "What errors occurred?",
              "Show the cause chain", "What evidence do we have?", "Tell me about TX90001"]:
        fail_503(gemini)
        body = chat(api_client, iid, q).json()
        assert body["reasoning_mode"] == "deterministic_fallback"
        assert _ID_RE.search(body["answer"]), f"no evidence cited for {q!r}"
        assert_ids_valid(iid, body)
        answers[q] = body["answer"].split("\n\n(Note:")[0]
    assert len(set(answers.values())) == len(answers), "each question gets its own answer"
    assert "Recovery signals observed" in answers["How did it recover?"]
    assert "Session leak identified" in answers["How did it recover?"] or "Restarting" in answers["How did it recover?"]
    assert answers["When did it start?"].startswith("The earliest abnormal event was [E00005]")
    assert "TX90001" in answers["Tell me about TX90001"]


@pytest.mark.parametrize("question, intent", [
    ("Why did the outage happen?", "cause"), ("What was the root cause?", "cause"),
    ("What caused the incident?", "cause"), ("What happened?", "timeline"), ("What failed?", "errors"),
    ("What services were affected?", "impact"), ("Which services were impacted?", "impact"),
    ("What happened first?", "start"), ("When did it start?", "start"), ("When did it recover?", "recovery"),
    ("What was the timeline?", "timeline"), ("What evidence do we have?", "evidence"),
    ("What errors occurred?", "errors"), ("What is the impact?", "impact"), ("How did it recover?", "recovery"),
    ("Unfold the incident", "timeline"), ("Show the cause chain", "chain"), ("Is the cache warm?", "generic"),
])
def test_intent_classification(question, intent):
    assert classify_intent(question) == intent


# --------------------------------------------------------------------------- #
# 9-10. Causal semantics
# --------------------------------------------------------------------------- #
def _answerer(name, question, graph_facts=()):
    events = LogEventParser().parse_file(str(FIX / name), "d")
    groups = EventProcessor().process(events)
    pack = EvidenceSelector().select(events, groups, question, budget_tokens=3000, mode="chat")
    by_id = {e.id: e for e in events}
    return DeterministicChatAnswerer(question, pack, groups, by_id.get, graph_facts=graph_facts), by_id


def _rel(source, target, rtype, basis, eid):
    return GraphRelationship(id=str(uuid.uuid4()), source=source, target=target, type=rtype,
                             investigation_id="i", basis=basis, evidence_event_ids=[eid])


def test_related_to_is_never_presented_as_cause():
    answerer, _ = _answerer("minimal_incident.log", "Why did the outage happen?", graph_facts=[
        _rel("Cache", "Checkout", RelationType.RELATED_TO, "explicit_text", "E00003"),
        _rel("Lag", "Cache", RelationType.PRECEDES, "temporal", "E00002"),
        _rel("Deploy", "Cache", RelationType.CAUSES, "co_occurrence", "E00001"),  # uncited -> not causal
    ])
    result = answerer.answer("server_error")
    assert "Causal links stated explicitly" not in result.answer
    assert "RELATED_TO" not in result.answer and "PRECEDES" not in result.answer
    root = [c for c in result.claims if c.text == "Root cause"]
    assert root and root[0].type == ClaimType.UNKNOWN


def test_explicit_causal_edge_is_reported_as_stated_in_logs():
    answerer, _ = _answerer("bracketed_multiline.txt", "What caused the incident?", graph_facts=[
        _rel("SocketTimeoutException", "SQLTransientConnectionException", RelationType.CAUSES, "explicit_text", "E00011"),
    ])
    result = answerer.answer("timeout")
    assert "Causal links stated explicitly in the logs: SocketTimeoutException CAUSES SQLTransientConnectionException [E00011]" in result.answer
    assert all(c.type != ClaimType.CONFIRMED for c in result.claims)


def test_sequence_is_not_asserted_as_causation():
    answerer, _ = _answerer("minimal_incident.log", "Why did the outage happen?")
    answer = answerer.answer("server_error").answer
    assert "does not conclusively establish which event caused the outage" in answer
    assert "temporal correlation" in answer
    assert not re.search(r"\b(caused|led to|resulted in)\b(?! the outage \(UNKNOWN\))", answer.replace(
        "which event caused the outage (UNKNOWN)", ""))


# --------------------------------------------------------------------------- #
# 12-15. Budgets, call policy, investigation, no stuck state
# --------------------------------------------------------------------------- #
def test_budgets_unchanged_and_chat_prompt_within_limits(api_client, gemini):
    s = get_settings()
    assert (s.MAX_GEMINI_CONTEXT_TOKENS, s.MAX_GEMINI_EVIDENCE_TOKENS, s.MAX_CHAT_EVIDENCE_TOKENS) == (6000, 4000, 3000)
    assert s.GEMINI_MAX_ATTEMPTS == 3
    iid = upload(api_client, "pool_leak_cascade.log")
    body = chat(api_client, iid, "Why did the outage happen?").json()
    call = gemini.calls[-1]
    assert estimate_tokens(call["prompt"]) + estimate_tokens(call["system"]) <= s.MAX_GEMINI_CONTEXT_TOKENS
    assert body["metrics"]["evidence_tokens_used"] <= s.MAX_CHAT_EVIDENCE_TOKENS


def test_investigation_unchanged_and_status_never_stuck(api_client, gemini):
    iid = upload(api_client, "pool_leak_cascade.log")
    report = api_client.post("/api/investigate", json={"investigation_id": iid}).json()
    assert report["metrics"]["gemini_logical_calls"] == 1 and len(gemini.calls) == 1
    for _ in range(3):
        fail_503(gemini)
        assert chat(api_client, iid, "What happened?").status_code == 200
    assert repo.get_investigation(iid).status.value == "COMPLETED"
    assert len(gemini.calls) == 1 + 3 * 3  # 1 investigation call; each chat = 1 logical call (3 bounded attempts)


def test_saturated_worker_pool_fails_fast(api_client, gemini, monkeypatch):
    import app.services.llm.gemini_client as gc

    busy = threading.BoundedSemaphore(1)
    busy.acquire()  # every worker "stuck" on a hung request
    monkeypatch.setattr(gc, "_INFLIGHT", busy)
    iid = upload(api_client, "pool_leak_cascade.log")
    started = time.perf_counter()
    body = chat(api_client, iid, "Which service failed first?").json()
    assert time.perf_counter() - started < 2
    assert body["reasoning_mode"] == "deterministic_fallback" and body["degradation_reason"] == "unavailable"
    assert body["metrics"]["gemini_api_attempts"] == 1 and gemini.calls == []


def test_internal_chat_error_is_readable(api_client, gemini, monkeypatch):
    from app.core.dependencies import get_investigation_agent

    iid = upload(api_client, "minimal_incident.log")
    monkeypatch.setattr(get_investigation_agent(), "ask", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    r = chat(api_client, iid, "What happened?")
    assert r.status_code == 500 and isinstance(r.json()["detail"], str) and "boom" not in r.json()["detail"]
