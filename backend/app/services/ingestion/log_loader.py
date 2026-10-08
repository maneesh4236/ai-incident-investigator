"""
Parses raw application/infra logs.

`LogLoader` (line-oriented, unchanged legacy API) and `LogEventParser`
(event-oriented): a log *event* is a header line that starts with a timestamp
plus every following continuation line (stack frames, `Caused by:`, wrapped
messages). Events are never split, and the original text of every event is
kept verbatim in `LogEvent.raw`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from app.core.logging_config import get_logger
from app.services.ingestion.events import LogEvent

logger = get_logger("ingestion.log_loader")

# Matches common timestamp formats:
# 2024-05-01T12:03:11Z, 2024-05-01 12:03:11, 12:03:11, May 01 12:03:11
_TIMESTAMP_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?"),
    re.compile(r"\b\d{2}:\d{2}:\d{2}\b"),
    re.compile(r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"),
]

_LEVEL_PATTERN = re.compile(r"\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b")


@dataclass
class LogLine:
    raw: str
    timestamp: Optional[str]
    level: Optional[str]
    line_number: int


class LogLoader:
    """Reads a log file and returns structured, timestamp-tagged lines (legacy API)."""

    def load(self, path: str) -> List[LogLine]:
        logger.info(f"Parsing log file: {path}")
        lines: List[LogLine] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, raw_line in enumerate(f, start=1):
                stripped = raw_line.rstrip("\r\n")
                if not stripped.strip():
                    continue
                lines.append(
                    LogLine(
                        raw=stripped,
                        timestamp=self._extract_timestamp(stripped),
                        level=self._extract_level(stripped),
                        line_number=idx,
                    )
                )
        logger.info(f"Parsed {len(lines)} non-empty log lines from {path}")
        return lines

    @staticmethod
    def _extract_timestamp(line: str) -> Optional[str]:
        for pattern in _TIMESTAMP_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(0)
        return None

    @staticmethod
    def _extract_level(line: str) -> Optional[str]:
        match = _LEVEL_PATTERN.search(line)
        return match.group(0).upper() if match else None

    def to_text(self, lines: List[LogLine]) -> str:
        """Recombine parsed lines into plain text for downstream chunking."""
        return "\n".join(line.raw for line in lines)


# --------------------------------------------------------------------------- #
# Event-oriented parsing
# --------------------------------------------------------------------------- #
_HEADER_TS = [
    re.compile(r"^\[?(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\]?"),
    re.compile(r"^\[?(?P<ts>\d{4}/\d{2}/\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]?"),
    re.compile(r"^\[?(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\]?"),
    re.compile(r"^\[?(?P<ts>\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]?(?=\s)"),
]

_LEVEL_ALIASES = {
    "TRACE": "TRACE", "DEBUG": "DEBUG", "INFO": "INFO", "NOTICE": "INFO",
    "WARN": "WARN", "WARNING": "WARN", "ERROR": "ERROR", "ERR": "ERROR",
    "FATAL": "CRITICAL", "CRITICAL": "CRITICAL", "SEVERE": "CRITICAL", "PANIC": "CRITICAL",
}
_TOKEN_RE = re.compile(r"\S+")
_SERVICE_TOKEN_RE = re.compile(r"^[A-Za-z][\w.\-]*[A-Za-z0-9]$")
_ID_LIKE_RE = re.compile(r"^(req|txn|trace|span|corr|request|correlation)[-_:=]?[\w-]*$|^[0-9a-fA-F-]{8,}$", re.I)
_SERVICE_KV_RE = re.compile(r"\b(?:service|svc|app)[=:]\s*([\w.\-]+)")

_ID_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("request_id", re.compile(r"\b(req-[\w-]+)")),
    ("request_id", re.compile(r"request[_-]?id[=:]\s*([\w-]+)", re.I)),
    ("txn_id", re.compile(r"\b(txn-[\w-]+)")),
    ("txn_id", re.compile(r"\btxn(?:Id)?[=:]\s*([\w-]+)", re.I)),
    ("trace_id", re.compile(r"trace[_-]?id[=:]\s*([\w-]+)", re.I)),
    ("correlation_id", re.compile(r"correlation[_-]?id[=:]\s*([\w-]+)", re.I)),
]
_STATUS_RE = re.compile(r"\bHTTP[ /]?(?:1\.[01] )?([1-5]\d\d)\b|\bstatus(?:_code)?[=: ]\s*([1-5]\d\d)\b", re.I)
_EXCEPTION_RE = re.compile(r"\b((?:[A-Za-z_$][\w$]*\.)*[A-Za-z_$][\w$]*(?:Exception|Error))\b")
_STACK_LINE_RE = re.compile(
    r"^\s+at\s|^\s*Caused by:|^\s*\.\.\. \d+ more|^Traceback \(most recent call last\)|"
    r"^\s+File \".*\", line \d+|^\s*(?:[\w$]+\.)+[\w$]*(?:Exception|Error)\b"
)
_METRIC_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    ("pool_pct", re.compile(r"pool[_ ]usage(?:\s*[=:]\s*|\s+at\s+)(\d+(?:\.\d+)?)\s*%", re.I), 1.0),
    ("error_rate_pct", re.compile(r"error rate[^0-9\n]{0,25}(\d+(?:\.\d+)?)\s*%", re.I), 1.0),
    ("latency_ms", re.compile(r"(\d+(?:\.\d+)?)\s*ms\b"), 1.0),
    ("latency_ms", re.compile(r"latency[^0-9\n]{0,30}(\d+(?:\.\d+)?)\s*s\b", re.I), 1000.0),
    ("active", re.compile(r"\bactive=(\d+)", re.I), 1.0),
    ("idle", re.compile(r"\bidle=(\d+)", re.I), 1.0),
    ("total", re.compile(r"\btotal=(\d+)", re.I), 1.0),
    ("active", re.compile(r"connections:\s*(\d+)\s*/\s*\d+", re.I), 1.0),
    ("total", re.compile(r"connections:\s*\d+\s*/\s*(\d+)", re.I), 1.0),
    ("active", re.compile(r"connections reduced to (\d+)", re.I), 1.0),
    ("sessions", re.compile(r"sessions\s*=\s*(\d+)", re.I), 1.0),
    ("threshold", re.compile(r"threshold\s*=\s*(\d+)", re.I), 1.0),
]

_SNIFF_LINES = 2000
_MIN_TIMESTAMPED_RATIO = 0.2
MAX_CONTINUATION_LINES = 2000


def _match_header(line: str) -> Optional[re.Match]:
    for pattern in _HEADER_TS:
        match = pattern.match(line)
        if match:
            return match
    return None


def parse_timestamp(raw: str) -> Optional[datetime]:
    text = raw.strip().strip("[]")
    try:
        if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", text):
            iso = text.replace(",", ".").replace("/", "-")
            if iso.endswith("Z"):
                iso = iso[:-1] + "+00:00"
            dt = datetime.fromisoformat(iso)
        elif re.match(r"^[A-Z][a-z]{2}\s", text):
            dt = datetime.strptime("1900 " + " ".join(text.split()), "%Y %b %d %H:%M:%S")
        else:
            main, _, frac = text.replace(",", ".").partition(".")
            dt = datetime.strptime("1900-01-01 " + main, "%Y-%m-%d %H:%M:%S")
            if frac:
                dt = dt.replace(microsecond=int((frac + "000000")[:6]))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class LogEventParser:
    """Streams a log file into complete `LogEvent`s (multiline-aware)."""

    def parse_file(self, path: str, document_id: str, id_offset: int = 0) -> List[LogEvent]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            sample = [l.rstrip("\r\n") for l in islice(f, _SNIFF_LINES * 3)]
        multiline_mode = self._timestamp_ratio(sample) >= _MIN_TIMESTAMPED_RATIO
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            events = list(self._parse_lines(f, document_id, id_offset, multiline_mode))
        logger.info(
            f"Parsed {len(events)} events ({sum(1 for e in events if e.is_multiline)} multiline) "
            f"from {path} (multiline_mode={multiline_mode})"
        )
        return events

    def parse_text(self, text: str, document_id: str = "doc", id_offset: int = 0) -> List[LogEvent]:
        lines = text.splitlines()
        multiline_mode = self._timestamp_ratio(lines[: _SNIFF_LINES * 3]) >= _MIN_TIMESTAMPED_RATIO
        return list(self._parse_lines(lines, document_id, id_offset, multiline_mode))

    @staticmethod
    def looks_like_log(path: str) -> bool:
        """True when >= 60% of the first non-empty lines start with a timestamp."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = [l for l in (x.rstrip("\r\n") for x in islice(f, 600)) if l.strip()][:200]
        except OSError:
            return False
        if not lines:
            return False
        return sum(1 for l in lines if _match_header(l)) / len(lines) >= 0.6

    @staticmethod
    def _timestamp_ratio(lines: List[str]) -> float:
        non_empty = [l for l in lines if l.strip()][:_SNIFF_LINES]
        if not non_empty:
            return 0.0
        return sum(1 for l in non_empty if _match_header(l)) / len(non_empty)

    # ------------------------------------------------------------------ #
    def _parse_lines(
        self, lines: Iterable[str], document_id: str, id_offset: int, multiline_mode: bool
    ) -> Iterator[LogEvent]:
        current: Optional[dict] = None
        seq = id_offset
        last_ts: Optional[datetime] = None

        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            header = _match_header(line)
            is_new = header is not None or not multiline_mode or current is None
            if not is_new and current is not None and len(current["lines"]) > MAX_CONTINUATION_LINES:
                is_new = True  # pathological continuation run: bounded, flagged via line numbers
            if is_new:
                if current is not None:
                    event = self._finish(current, seq, document_id, last_ts)
                    last_ts = event.ts or last_ts
                    seq += 1
                    yield event
                current = {"lines": [line], "line_start": line_no, "line_end": line_no, "header": header}
            else:
                current["lines"].append(line)
                current["line_end"] = line_no

        if current is not None:
            yield self._finish(current, seq, document_id, last_ts)

    def _finish(self, cur: dict, seq: int, document_id: str, last_ts: Optional[datetime]) -> LogEvent:
        lines: List[str] = cur["lines"]
        first = lines[0]
        header: Optional[re.Match] = cur["header"]

        ts_raw: Optional[str] = None
        rest = first
        if header is not None:
            ts_raw = header.group("ts")
            rest = first[header.end():]
        else:
            loose = LogLoader._extract_timestamp(first)
            ts_raw = loose
        ts = parse_timestamp(ts_raw) if ts_raw else None

        level, service, message, bracket_ids = self._split_header(rest)
        continuation = lines[1:]
        raw = "\n".join(lines)

        ids: Dict[str, str] = {}
        for name, pattern in _ID_PATTERNS:
            if name in ids:
                continue
            match = pattern.search(first)
            if match:
                ids[name] = match.group(1)
        for value in bracket_ids:
            ids.setdefault("request_id" if value.lower().startswith("req") else "txn_id" if value.lower().startswith("txn") else "trace_id", value)

        status_code = None
        status_match = _STATUS_RE.search(first)
        if status_match:
            status_code = int(status_match.group(1) or status_match.group(2))

        exception_type = None
        for candidate_line in [message] + continuation:
            match = _EXCEPTION_RE.search(candidate_line)
            if match:
                exception_type = match.group(1)
                break

        has_stack = any(_STACK_LINE_RE.search(l) for l in continuation)

        metrics: Dict[str, float] = {}
        for name, pattern, factor in _METRIC_PATTERNS:
            if name in metrics:
                continue
            match = pattern.search(message)
            if match:
                metrics[name] = float(match.group(1)) * factor

        return LogEvent(
            id=f"E{seq + 1:05d}",
            seq=seq,
            document_id=document_id,
            line_start=cur["line_start"],
            line_end=cur["line_end"],
            raw=raw,
            message=message,
            ts_raw=ts_raw,
            ts=ts or last_ts,
            level=level,
            service=service,
            exception_type=exception_type,
            has_stack_trace=has_stack,
            continuation_lines=len(continuation),
            status_code=status_code,
            ids=ids or None,
            metrics=metrics or None,
        )

    @staticmethod
    def _split_header(rest: str) -> Tuple[Optional[str], Optional[str], str, List[str]]:
        """Extracts level, service and message from the text after the timestamp."""
        tokens = list(_TOKEN_RE.finditer(rest))
        level = None
        level_idx = -1
        for i, tok in enumerate(tokens[:4]):
            cleaned = tok.group(0).strip("[]():,|-").upper()
            if cleaned in _LEVEL_ALIASES:
                level, level_idx = _LEVEL_ALIASES[cleaned], i
                break

        service = None
        bracket_ids: List[str] = []
        msg_start_idx = 0
        if level_idx >= 0:
            msg_start_idx = level_idx + 1
            if level_idx >= 1:
                candidate = tokens[level_idx - 1].group(0).strip("[]():,|")
                if _SERVICE_TOKEN_RE.match(candidate) and candidate.upper() not in _LEVEL_ALIASES:
                    service = candidate
            # Bracketed tokens right after the level: [service] [req-id] ...
            j = msg_start_idx
            while j < len(tokens):
                tok = tokens[j].group(0)
                if not (tok.startswith("[") and tok.endswith("]") and len(tok) > 2):
                    break
                inner = tok[1:-1]
                if _ID_LIKE_RE.match(inner):
                    bracket_ids.append(inner)
                elif service is None and _SERVICE_TOKEN_RE.match(inner):
                    service = inner
                j += 1
            msg_start_idx = j
        else:
            legacy = _LEVEL_PATTERN.search(rest)
            if legacy:
                level = _LEVEL_ALIASES.get(legacy.group(0).upper())

        if service is None:
            kv = _SERVICE_KV_RE.search(rest)
            if kv:
                service = kv.group(1)

        message = rest[tokens[msg_start_idx].start():].strip() if msg_start_idx < len(tokens) else rest.strip()
        if not message:
            message = rest.strip()
        return level, service, message, bracket_ids
