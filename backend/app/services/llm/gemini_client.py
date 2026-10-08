"""
Thin wrapper around Google's Gemini SDK (`google-genai`).

Centralizing all LLM calls here means every other service talks to a single,
testable interface instead of importing the SDK directly.

Guarantees provided by `call()` (it never raises):
  * Budget guard  - a prompt whose estimated size exceeds
    MAX_GEMINI_CONTEXT_TOKENS is never sent.
  * Call limit    - the current request scope's `max_gemini_calls`
    (upload 0, investigate 1, chat 1) is enforced here.
  * Timeout       - google-genai 0.3.0 hard-codes `timeout=None`, so each
    attempt runs in a worker thread with a deadline.
  * Bounded retry - only 429 / 5xx / network / (once) timeout are retried, at
    most GEMINI_MAX_ATTEMPTS attempts and GEMINI_MAX_TOTAL_SECONDS wall time.
    Daily-quota 429s and other 4xx are not retried.
  * Metrics       - attempts, retries, errors by kind, latency and actual
    token usage (usage_metadata) are recorded on the request scope.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.metrics import current_metrics
from app.services.reasoning.token_budget import estimate_tokens

logger = get_logger("llm.gemini_client")

# Shared pool used only to impose a deadline on the blocking SDK call. A timed
# out attempt may keep its thread until the socket returns (the SDK has no
# socket timeout); the caller stops waiting immediately. `_INFLIGHT` tracks
# busy workers so that, if all of them are stuck on hung requests, new calls
# fail fast ("unavailable") instead of queueing behind them until timeout.
_MAX_INFLIGHT = 8
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_INFLIGHT, thread_name_prefix="gemini-call")
_INFLIGHT = threading.BoundedSemaphore(_MAX_INFLIGHT)

_RETRY_DELAY_RE = re.compile(r'retryDelay["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)s')

# Masked from diagnostic log lines: (pattern, replacement).
_SECRET_PATTERNS = [
    # header / field assignments: "x-goog-api-key: ...", "api_key=...", "Authorization: Bearer ..."
    (re.compile(r"(?i)\b(x-goog-api-key|authorization|api[_-]?key|access[_-]?token|password|secret)"
                r"([\"']?\s*[:=]\s*[\"']?)(?:bearer\s+)?[^\s\"',}&]+"), r"\1\2[REDACTED]"),
    (re.compile(r"(?i)([?&]key=)[^&\s\"']+"), r"\1[REDACTED]"),       # ?key=... in URLs
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"), "Bearer [REDACTED]"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED]"),            # Google API key shape
    # long opaque tokens (letters + digits, 35+ chars)
    (re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{35,}\b"), "[REDACTED]"),
]


def _redact(text: str, secrets: Optional[list] = None) -> str:
    """Masks configured secret values and secret-shaped substrings."""
    for secret in secrets or []:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "[REDACTED]")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


_RETRYABLE = {"rate_limited", "server_error", "network", "timeout"}
_MIN_ATTEMPT_SECONDS = 1.0  # do not start an attempt with less time than this left


@dataclass
class GeminiResult:
    ok: bool
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error_kind: Optional[str] = None
    error_message: Optional[str] = None
    attempts: int = 0
    latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_prompt_tokens: int = 0


class _AttemptError(Exception):
    def __init__(self, kind: str, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after


class GeminiClient:
    def __init__(self, sleep: Callable[[float], None] = time.sleep):
        settings = get_settings()
        self._settings = settings
        self._model_name = settings.GEMINI_MODEL
        self._configured = bool(settings.GEMINI_API_KEY)
        self._sleep = sleep
        if self._configured:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        else:
            logger.warning(
                "GEMINI_API_KEY not set. GeminiClient will run in offline mode and "
                "callers will use their deterministic fallbacks."
            )
            self._client = None

    @property
    def is_configured(self) -> bool:
        return self._configured

    # ------------------------------------------------------------------ #
    # Core entry point
    # ------------------------------------------------------------------ #
    def call(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        *,
        purpose: str = "generic",
        json_mode: bool = True,
        max_output_tokens: Optional[int] = None,
        deadline_seconds: Optional[float] = None,
    ) -> GeminiResult:
        """`deadline_seconds` optionally tightens GEMINI_MAX_TOTAL_SECONDS for this call
        (used by chat so a struggling Gemini falls back quickly). The total limit is
        enforced inside attempts too: each attempt's timeout is capped by the time left."""
        settings = self._settings
        total_limit = settings.GEMINI_MAX_TOTAL_SECONDS
        if deadline_seconds is not None:
            total_limit = min(total_limit, deadline_seconds)
        metrics = current_metrics()
        estimated = estimate_tokens(prompt) + estimate_tokens(system_instruction or "")

        if not self._configured:
            return self._refuse("not_configured", "GEMINI_API_KEY not set", estimated, metrics, count=False)

        if estimated > settings.MAX_GEMINI_CONTEXT_TOKENS:
            logger.error(
                f"Refusing Gemini call purpose={purpose}: estimated {estimated} tokens "
                f"> MAX_GEMINI_CONTEXT_TOKENS={settings.MAX_GEMINI_CONTEXT_TOKENS}"
            )
            return self._refuse("budget_exceeded", "prompt exceeds context budget", estimated, metrics)

        if metrics is not None and not metrics.try_reserve_gemini_call():
            logger.error(
                f"Refusing Gemini call purpose={purpose}: request scope '{metrics.name}' "
                f"allows {metrics.max_gemini_calls} call(s)"
            )
            return self._refuse("call_limit_exceeded", "per-request Gemini call limit reached", estimated, metrics)
        if metrics is not None:
            metrics.incr("gemini_estimated_prompt_tokens", estimated)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=settings.GEMINI_TEMPERATURE,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if json_mode else None,
        )

        started = time.perf_counter()
        attempts = 0
        timeouts = 0
        last_error: Optional[_AttemptError] = None

        while attempts < settings.GEMINI_MAX_ATTEMPTS:
            remaining = total_limit - (time.perf_counter() - started)
            if attempts > 0 and remaining < _MIN_ATTEMPT_SECONDS:
                break
            attempt_timeout = min(settings.GEMINI_TIMEOUT_SECONDS, max(remaining, _MIN_ATTEMPT_SECONDS))
            attempts += 1
            if metrics is not None:
                metrics.incr("gemini_api_attempts")
                if attempts > 1:
                    metrics.incr("gemini_retries")
            attempt_start = time.perf_counter()
            try:
                response = self._attempt(prompt, config, attempt_timeout)
                latency = (time.perf_counter() - attempt_start) * 1000
                result = self._build_result(response, attempts, latency, estimated, json_mode)
                self._record_attempt(metrics, purpose, attempts, "ok" if result.ok else result.error_kind, latency, result)
                if result.ok:
                    if metrics is not None:
                        metrics.incr("gemini_successes")
                else:
                    self._record_failure(metrics, result.error_kind)
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result
            except _AttemptError as exc:
                latency = (time.perf_counter() - attempt_start) * 1000
                last_error = exc
                self._record_attempt(metrics, purpose, attempts, exc.kind, latency, None)
                if exc.kind == "timeout":
                    timeouts += 1
                retryable = exc.kind in _RETRYABLE and not (exc.kind == "timeout" and timeouts > 1)
                if not retryable or attempts >= settings.GEMINI_MAX_ATTEMPTS:
                    break
                elapsed = time.perf_counter() - started
                remaining = total_limit - elapsed
                delay = self._backoff(attempts, exc.retry_after)
                if delay + _MIN_ATTEMPT_SECONDS >= remaining:
                    break
                self._sleep(delay)

        kind = last_error.kind if last_error else "unexpected"
        self._record_failure(metrics, kind)
        return GeminiResult(
            ok=False,
            error_kind=kind,
            error_message=str(last_error) if last_error else None,
            attempts=attempts,
            latency_ms=(time.perf_counter() - started) * 1000,
            estimated_prompt_tokens=estimated,
        )

    # ------------------------------------------------------------------ #
    # Backward-compatible helpers (never raise)
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        result = self.call(prompt, system_instruction, purpose="generate", json_mode=False)
        return result.text if result.ok else ""

    def generate_json(self, prompt: str, system_instruction: Optional[str] = None) -> Dict[str, Any]:
        result = self.call(prompt, system_instruction, purpose="generate_json", json_mode=True)
        return result.data if result.ok else {}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _attempt(self, prompt: str, config: types.GenerateContentConfig, timeout: Optional[float] = None):
        timeout = timeout or self._settings.GEMINI_TIMEOUT_SECONDS
        if not _INFLIGHT.acquire(blocking=False):
            # Every worker is still blocked on an earlier (hung) request; queueing
            # would only burn the whole timeout. Fail fast; callers fall back.
            raise _AttemptError("unavailable", "all Gemini worker threads are busy with unfinished requests")
        try:
            future = _EXECUTOR.submit(
                self._client.models.generate_content, model=self._model_name, contents=prompt, config=config
            )
        except Exception:
            _INFLIGHT.release()
            raise
        future.add_done_callback(lambda _f: _INFLIGHT.release())
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            raise _AttemptError("timeout", f"no response within {timeout:.0f}s")
        except genai_errors.APIError as exc:
            attempt_error = self._classify_api_error(exc)
            self._log_error_diagnostics(exc, attempt_error.kind)
            raise attempt_error
        except Exception as exc:  # network / transport / SDK errors
            attempt_error = self._classify_other_error(exc)
            self._log_error_diagnostics(exc, attempt_error.kind)
            raise attempt_error

    def _log_error_diagnostics(self, exc: Exception, kind: str) -> None:
        """Logs the real API error behind a classified attempt failure, with secrets redacted.

        Never logs the API key, auth headers or any credential: the configured key
        values and key/token/password-shaped strings are masked before logging.
        """
        details = getattr(exc, "details", None)
        error_obj = details.get("error", details) if isinstance(details, dict) else {}
        if not isinstance(error_obj, dict):
            error_obj = {}
        reasons = [
            d.get("reason") for d in (error_obj.get("details") or []) if isinstance(d, dict) and d.get("reason")
        ]
        message = getattr(exc, "message", None) or error_obj.get("message") or str(exc)
        secrets = [self._settings.GEMINI_API_KEY, os.environ.get("GOOGLE_API_KEY", "")]
        logger.warning(
            "gemini_api_error kind={} exception={} http_status={} api_code={} api_status={} reason={} "
            "message=\"{}\" model={} GEMINI_API_KEY_present={} GOOGLE_API_KEY_present={}",
            kind,
            type(exc).__name__,
            getattr(exc, "code", None),
            error_obj.get("code"),
            _redact(str(getattr(exc, "status", None) or error_obj.get("status")), secrets),
            _redact(",".join(reasons) or "none", secrets),
            _redact(str(message), secrets)[:500],
            self._model_name,
            "YES" if self._settings.GEMINI_API_KEY else "NO",
            "YES" if os.environ.get("GOOGLE_API_KEY") else "NO",
        )

    @staticmethod
    def _classify_api_error(exc: "genai_errors.APIError") -> _AttemptError:
        code = getattr(exc, "code", None) or 0
        details = json.dumps(getattr(exc, "details", None), default=str) if getattr(exc, "details", None) else str(exc)
        if code == 429:
            if re.search(r"per ?day|PerDay", details, re.IGNORECASE):
                return _AttemptError("quota_exhausted", f"429 daily quota exhausted: {exc}")
            match = _RETRY_DELAY_RE.search(details)
            return _AttemptError("rate_limited", f"429: {exc}", float(match.group(1)) if match else None)
        if 500 <= code < 600:
            return _AttemptError("server_error", f"{code}: {exc}")
        return _AttemptError("client_error", f"{code}: {exc}")

    @staticmethod
    def _classify_other_error(exc: Exception) -> _AttemptError:
        name = type(exc).__name__.lower()
        text = str(exc)
        if re.match(r"^\s*429\b", text) or "resource_exhausted" in text.lower():
            return _AttemptError("rate_limited", text)
        if re.match(r"^\s*5\d\d\b", text):
            return _AttemptError("server_error", text)
        if any(k in name for k in ("connection", "timeout", "chunkedencoding", "protocol", "remotedisconnected")):
            return _AttemptError("network", f"{type(exc).__name__}: {text}")
        if isinstance(exc, (ConnectionError, OSError)):
            return _AttemptError("network", f"{type(exc).__name__}: {text}")
        return _AttemptError("unexpected", f"{type(exc).__name__}: {text}")

    def _backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        s = self._settings
        if retry_after is not None:
            return max(retry_after, 0.0)
        base = min(s.GEMINI_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), s.GEMINI_BACKOFF_MAX_SECONDS)
        return base * (0.5 + random.random() / 2)

    def _build_result(self, response, attempts: int, latency: float, estimated: int, json_mode: bool) -> GeminiResult:
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        try:
            text = response.text or ""
        except Exception:  # SDK raises on some blocked responses
            text = ""
        base = dict(attempts=attempts, latency_ms=latency, prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens, estimated_prompt_tokens=estimated)
        if not text.strip():
            feedback = getattr(response, "prompt_feedback", None)
            kind = "blocked" if feedback is not None and getattr(feedback, "block_reason", None) else "empty"
            return GeminiResult(ok=False, error_kind=kind, error_message="empty response", **base)
        if not json_mode:
            return GeminiResult(ok=True, text=text, **base)
        data = parse_json_object(text)
        if data is None:
            return GeminiResult(ok=False, text=text, error_kind="invalid_json", error_message="unparseable JSON", **base)
        return GeminiResult(ok=True, text=text, data=data, **base)

    @staticmethod
    def _record_attempt(metrics, purpose: str, attempt: int, status: str, latency: float, result: Optional[GeminiResult]):
        prompt_tokens = result.prompt_tokens if result else None
        output_tokens = result.output_tokens if result else None
        if metrics is not None:
            with metrics._lock:
                metrics.gemini_latency_ms.append(latency)
            if prompt_tokens:
                metrics.incr("gemini_prompt_tokens", int(prompt_tokens))
            if output_tokens:
                metrics.incr("gemini_output_tokens", int(output_tokens))
        logger.info(
            f"gemini_attempt purpose={purpose} attempt={attempt} status={status} "
            f"latency_ms={latency:.0f} prompt_tokens={prompt_tokens} output_tokens={output_tokens}"
        )

    @staticmethod
    def _record_failure(metrics, kind: Optional[str]) -> None:
        if metrics is not None:
            metrics.incr("gemini_failures")
            metrics.record_error(kind or "unexpected")

    @staticmethod
    def _refuse(kind: str, message: str, estimated: int, metrics, count: bool = True) -> GeminiResult:
        if metrics is not None and count:
            metrics.record_error(kind)
        return GeminiResult(ok=False, error_kind=kind, error_message=message, estimated_prompt_tokens=estimated)


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Parses a JSON object out of a model response (tolerates code fences / prose)."""
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None
