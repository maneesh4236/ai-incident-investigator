"""
Per-request pipeline metrics and the Gemini call-limit guard.

A `RequestMetrics` object is bound to a contextvar for the duration of one
API request (`metrics_scope`). Services increment counters on whatever scope
is current, so call amplification is measured rather than guessed. The scope
also carries `max_gemini_calls`, which `GeminiClient` enforces: upload = 0,
investigate = 1, chat = 1.

`run_in_threadpool` / FastAPI sync routes copy the context, so the scope is
visible inside worker threads.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class RequestMetrics:
    name: str
    max_gemini_calls: Optional[int] = None  # None = unlimited (scripts/tests)

    # Gemini
    gemini_logical_calls: int = 0
    gemini_api_attempts: int = 0
    gemini_retries: int = 0
    gemini_successes: int = 0
    gemini_failures: int = 0
    gemini_errors: Dict[str, int] = field(default_factory=dict)
    gemini_latency_ms: List[float] = field(default_factory=list)
    gemini_prompt_tokens: int = 0  # actual, from usage_metadata
    gemini_output_tokens: int = 0  # actual, from usage_metadata
    gemini_estimated_prompt_tokens: int = 0
    gemini_calls_refused: int = 0

    # Budget / evidence
    context_budget_tokens: int = 0
    evidence_budget_tokens: int = 0
    evidence_tokens_used: int = 0
    prompt_tokens_estimated: int = 0

    # Pipeline
    parsed_events: int = 0
    multiline_events: int = 0
    templates: int = 0
    groups: int = 0
    selected_events: int = 0
    protected_events: int = 0
    omitted_events: int = 0
    qdrant_ops: int = 0
    qdrant_points: int = 0
    graph_ops: int = 0
    stage_ms: Dict[str, float] = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)

    def record_error(self, kind: str) -> None:
        with self._lock:
            self.gemini_errors[kind] = self.gemini_errors.get(kind, 0) + 1

    def try_reserve_gemini_call(self) -> bool:
        """Atomically reserve one logical Gemini call; False if over the limit."""
        with self._lock:
            if self.max_gemini_calls is not None and self.gemini_logical_calls >= self.max_gemini_calls:
                self.gemini_calls_refused += 1
                return False
            self.gemini_logical_calls += 1
            return True

    @contextmanager
    def stage(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            with self._lock:
                self.stage_ms[stage_name] = round(self.stage_ms.get(stage_name, 0.0) + elapsed, 2)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            data = {
                f.name: (dict(v) if isinstance(v := getattr(self, f.name), dict) else list(v) if isinstance(v, list) else v)
                for f in fields(self)
                if f.name != "_lock"
            }
        lat = self.gemini_latency_ms
        data["gemini_latency_ms_total"] = round(sum(lat), 2)
        data["gemini_latency_ms"] = [round(x, 2) for x in lat]
        data["gemini_total_tokens"] = self.gemini_prompt_tokens + self.gemini_output_tokens
        return data


_current: ContextVar[Optional[RequestMetrics]] = ContextVar("aetherlog_request_metrics", default=None)

# Process-wide totals, for GET /api/metrics.
_process_totals: Dict[str, Any] = {"requests": 0}
_process_lock = threading.Lock()


def current_metrics() -> Optional[RequestMetrics]:
    return _current.get()


@contextmanager
def metrics_scope(name: str, max_gemini_calls: Optional[int] = None) -> Iterator[RequestMetrics]:
    metrics = RequestMetrics(name=name, max_gemini_calls=max_gemini_calls)
    token = _current.set(metrics)
    start = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics.stage_ms["total"] = round((time.perf_counter() - start) * 1000, 2)
        _current.reset(token)
        _merge_into_process_totals(metrics)


def _merge_into_process_totals(metrics: RequestMetrics) -> None:
    numeric = {k: v for k, v in metrics.to_dict().items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    with _process_lock:
        _process_totals["requests"] += 1
        for key, value in numeric.items():
            if key in ("max_gemini_calls",):
                continue
            _process_totals[key] = _process_totals.get(key, 0) + value
        errors = _process_totals.setdefault("gemini_errors", {})
        for kind, count in metrics.gemini_errors.items():
            errors[kind] = errors.get(kind, 0) + count


def process_totals() -> Dict[str, Any]:
    with _process_lock:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _process_totals.items()}
