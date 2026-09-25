"""
Parses raw application/infra logs into normalized, timestamp-tagged lines.

Logs are treated differently from documents because RCA and timeline
reconstruction depend heavily on being able to recover a chronological
ordering of events, even across heterogeneous log formats.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from app.core.logging_config import get_logger

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
    """Reads a log file and returns structured, timestamp-tagged lines."""

    def load(self, path: str) -> List[LogLine]:
        logger.info(f"Parsing log file: {path}")
        lines: List[LogLine] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, raw_line in enumerate(f, start=1):
                stripped = raw_line.rstrip("\n")
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
