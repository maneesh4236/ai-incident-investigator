"""
Local token estimation and a hard budget accumulator.

Gemini's tokenizer is not available offline and `count_tokens` would itself
be an extra API call, so we estimate conservatively (chars / 3.0 by default;
log text measured at ~3.4 chars per word-piece) and calibrate against the
actual `usage_metadata.prompt_token_count` recorded after each call.
"""
from __future__ import annotations

import math

from app.core.config import get_settings

_PER_TEXT_OVERHEAD = 4


def estimate_tokens(text: str, chars_per_token: float | None = None) -> int:
    if not text:
        return 0
    cpt = chars_per_token or get_settings().TOKEN_ESTIMATE_CHARS_PER_TOKEN
    return math.ceil(len(text) / cpt) + _PER_TEXT_OVERHEAD


class TokenBudget:
    """Accumulates text against a hard limit. `add` refuses anything that would overflow."""

    def __init__(self, limit: int, chars_per_token: float | None = None):
        if limit < 0:
            raise ValueError("budget limit must be >= 0")
        self.limit = limit
        self.used = 0
        self._cpt = chars_per_token

    def cost(self, text: str) -> int:
        return estimate_tokens(text, self._cpt)

    def fits(self, text: str) -> bool:
        return self.used + self.cost(text) <= self.limit

    def add(self, text: str) -> bool:
        cost = self.cost(text)
        if self.used + cost > self.limit:
            return False
        self.used += cost
        return True

    def release(self, text: str) -> None:
        self.used = max(0, self.used - self.cost(text))

    @property
    def remaining(self) -> int:
        return self.limit - self.used
