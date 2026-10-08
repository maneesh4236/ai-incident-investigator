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
import random
import re
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
# out attempt may keep its thread until the socket returns; the caller stops
# waiting immediately.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini-call")

_RETRY_DELAY_RE = re.compile(r'retryDelay["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)s')
_RETRYABLE = {"rate_limited", "server_error", "network", "timeout"}


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
    ) -> GeminiResult:
        settings = self._settings
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
            attempts += 1
            if metrics is not None:
                metrics.incr("gemini_api_attempts")
                if attempts > 1:
                    metrics.incr("gemini_retries")
            attempt_start = time.perf_counter()
            try:
                response = self._attempt(prompt, config)
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
                remaining = settings.GEMINI_MAX_TOTAL_SECONDS - elapsed
                delay = self._backoff(attempts, exc.retry_after)
                if delay >= remaining:
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
    def _attempt(self, prompt: str, config: types.GenerateContentConfig):
        future = _EXECUTOR.submit(
            self._client.models.generate_content, model=self._model_name, contents=prompt, config=config
        )
        try:
            return future.result(timeout=self._settings.GEMINI_TIMEOUT_SECONDS)
        except FutureTimeout:
            future.cancel()
            raise _AttemptError("timeout", f"no response within {self._settings.GEMINI_TIMEOUT_SECONDS}s")
        except genai_errors.APIError as exc:
            raise self._classify_api_error(exc)
        except Exception as exc:  # network / transport / SDK errors
            raise self._classify_other_error(exc)

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
