import time

import pytest

from app.core.config import get_settings
from app.core.metrics import metrics_scope
from app.services.llm.gemini_client import GeminiClient
from app.tests.conftest import FakeResponse, gemini_error


@pytest.fixture
def client():
    c = GeminiClient(sleep=lambda s: None)
    c._client.script.clear()
    c._client.calls.clear()
    return c


def test_429_then_success_counts_attempts_and_retries(client):
    client._client.script.extend([gemini_error(429), gemini_error(429), {"ok": 1}])
    with metrics_scope("t", max_gemini_calls=1) as m:
        result = client.call("p" * 400, "s", purpose="t")
    assert result.ok and result.data == {"ok": 1} and result.attempts == 3
    assert m.gemini_logical_calls == 1 and m.gemini_api_attempts == 3 and m.gemini_retries == 2
    assert m.gemini_successes == 1 and m.gemini_failures == 0
    assert m.gemini_prompt_tokens > 0


@pytest.mark.parametrize("code, kind, attempts", [(429, "rate_limited", 3), (503, "server_error", 3),
                                                  (500, "server_error", 3), (400, "client_error", 1),
                                                  (403, "client_error", 1)])
def test_persistent_errors_are_bounded_and_never_raise(client, code, kind, attempts):
    client._client.script.extend([gemini_error(code) for _ in range(5)])
    with metrics_scope("t") as m:
        result = client.call("p", purpose="t")
    assert not result.ok and result.error_kind == kind and result.attempts == attempts
    assert m.gemini_errors == {kind: 1}


def test_daily_quota_is_not_retried(client):
    client._client.script.append(gemini_error(429, "RESOURCE_EXHAUSTED", {"quotaId": "GenerateRequestsPerDayPerProject"}))
    result = client.call("p")
    assert result.error_kind == "quota_exhausted" and result.attempts == 1


def test_retry_delay_from_server_is_honoured():
    slept = []
    c = GeminiClient(sleep=slept.append)
    c._client.script.clear()
    c._client.script.extend([gemini_error(429, "x", {"error": {"details": [{"retryDelay": "2s"}]}}), {"ok": 1}])
    assert c.call("p").ok
    assert slept == [2.0]


def test_timeout_is_enforced_and_retried_once(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "GEMINI_TIMEOUT_SECONDS", 0.1)

    def slow(contents, config):
        time.sleep(0.5)
        return {"late": True}

    client._client.script.extend([slow, slow, slow])
    started = time.perf_counter()
    result = client.call("p")
    assert time.perf_counter() - started < 0.6
    assert result.error_kind == "timeout" and result.attempts == 2


def test_network_error_is_retried(client):
    client._client.script.extend([ConnectionError("reset"), {"ok": 1}])
    assert client.call("p").attempts == 2


def test_empty_and_invalid_json(client):
    client._client.script.append(FakeResponse(""))
    assert client.call("p").error_kind == "empty"
    client._client.script.append("definitely not json")
    assert client.call("p").error_kind == "invalid_json"
    client._client.script.append("```json\n{\"a\": 1}\n```")
    assert client.call("p").data == {"a": 1}


def test_budget_guard_sends_nothing(client):
    huge = "x" * (get_settings().MAX_GEMINI_CONTEXT_TOKENS * 4)
    with metrics_scope("t") as m:
        result = client.call(huge)
    assert result.error_kind == "budget_exceeded" and result.attempts == 0
    assert client._client.calls == [] and m.gemini_api_attempts == 0


def test_call_limit_enforced_per_scope(client):
    with metrics_scope("upload", max_gemini_calls=0) as m:
        result = client.call("p")
    assert result.error_kind == "call_limit_exceeded" and client._client.calls == []
    assert m.gemini_calls_refused == 1
    with metrics_scope("investigate", max_gemini_calls=1):
        assert client.call("p").ok
        assert client.call("p").error_kind == "call_limit_exceeded"
    assert len(client._client.calls) == 1


def test_legacy_helpers_never_raise(client):
    client._client.script.extend([gemini_error(503)] * 6)
    assert client.generate_json("p") == {}
    assert client.generate("p") == ""


def test_not_configured(monkeypatch):
    monkeypatch.setattr(get_settings(), "GEMINI_API_KEY", "")
    c = GeminiClient()
    assert not c.is_configured and c.call("p").error_kind == "not_configured"
