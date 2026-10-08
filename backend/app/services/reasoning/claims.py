"""
Deterministic validation of Gemini claims against the evidence actually sent.

Rules (applied after every Gemini response, RCA and chat alike):
  * Citations must be ids that were present in the prompt; others are
    dropped and counted as rejected (invented ids never survive).
  * OBSERVED / CONFIRMED with no valid citation       -> UNKNOWN.
  * LIKELY with no valid citation                     -> INFERRED.
  * CONFIRMED requires a cited event that is a diagnosis statement (DIAGNOSIS
    tag: "root cause", "identified", "confirmed", "caused by", "due to") or
    carries an explicit `Caused by:` chain          -> otherwise LIKELY.
  * A claim asserting exhaustion must cite an event whose text says
    "exhaust..." (a metric below 100% is not exhaustion) -> otherwise INFERRED.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from app.models.schemas import Claim, ClaimType
from app.services.ingestion.events import LogEvent

_ID_RE = re.compile(r"^\s*\[?\s*([ED])\s*0*(\d+)\s*\]?\s*$", re.I)
_EXHAUST_CLAIM_RE = re.compile(r"\bexhaust", re.I)
_EXHAUST_EVIDENCE_RE = re.compile(r"\bexhaust", re.I)
_CAUSED_BY_RE = re.compile(r"Caused by:", re.I)

CONFIDENCE_WORDS = {"very high": 0.9, "high": 0.8, "medium": 0.5, "moderate": 0.5, "low": 0.3, "very low": 0.15}
CONFIDENCE_CAPS = {
    ClaimType.CONFIRMED: 1.0,
    ClaimType.LIKELY: 0.75,
    ClaimType.INFERRED: 0.6,
    ClaimType.OBSERVED: 0.75,
    ClaimType.UNKNOWN: 0.3,
}


def normalize_id(raw: Any) -> Optional[str]:
    match = _ID_RE.match(str(raw))
    if not match:
        return None
    return f"{match.group(1).upper()}{int(match.group(2)):05d}"


def coerce_claim_type(raw: Any) -> ClaimType:
    value = str(raw or "").strip().upper()
    return ClaimType[value] if value in ClaimType.__members__ else ClaimType.UNKNOWN


def coerce_confidence(raw: Any, default: float = 0.5) -> float:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
    else:
        text = str(raw or "").strip().lower().rstrip("%")
        if text in CONFIDENCE_WORDS:
            value = CONFIDENCE_WORDS[text]
        else:
            try:
                value = float(text)
            except ValueError:
                value = default
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def coerce_str(raw: Any, keys: Tuple[str, ...] = ("text", "statement", "step", "action", "name", "title")) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for key in keys:
            if isinstance(raw.get(key), str) and raw[key].strip():
                return raw[key].strip()
    return str(raw).strip() if raw is not None else ""


def coerce_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [s for s in (coerce_str(item) for item in raw) if s]


class ClaimValidator:
    def __init__(self, valid_ids: Set[str], event_lookup: Callable[[str], Optional[LogEvent]]):
        self.valid_ids = set(valid_ids)
        self.lookup = event_lookup
        self.rejected_ids: List[str] = []
        self.downgrades: List[str] = []

    def citations(self, raw_ids: Any) -> List[str]:
        if raw_ids is None:
            return []
        if isinstance(raw_ids, (str, int)):
            raw_ids = re.split(r"[,\s]+", str(raw_ids))
        out: List[str] = []
        for raw in raw_ids if isinstance(raw_ids, list) else []:
            norm = normalize_id(raw)
            if norm and norm in self.valid_ids:
                if norm not in out:
                    out.append(norm)
            elif str(raw).strip():
                self.rejected_ids.append(str(raw).strip())
        return out

    def validate(self, text: str, raw_type: Any, raw_ids: Any) -> Claim:
        claim_type = coerce_claim_type(raw_type)
        ids = self.citations(raw_ids)
        original = claim_type

        if not ids:
            if claim_type in (ClaimType.OBSERVED, ClaimType.CONFIRMED):
                claim_type = ClaimType.UNKNOWN
            elif claim_type == ClaimType.LIKELY:
                claim_type = ClaimType.INFERRED

        if claim_type == ClaimType.CONFIRMED and not self._has_confirmation(ids):
            claim_type = ClaimType.LIKELY

        if _EXHAUST_CLAIM_RE.search(text or "") and claim_type in (ClaimType.OBSERVED, ClaimType.CONFIRMED, ClaimType.LIKELY):
            if not self._cites_text(ids, _EXHAUST_EVIDENCE_RE):
                claim_type = ClaimType.INFERRED

        if claim_type != original:
            self.downgrades.append(f"{original.value}->{claim_type.value}: {text[:80]}")
        return Claim(text=text, type=claim_type, evidence_ids=ids, citations_valid=bool(ids))

    def validate_items(self, raw_items: Any, text_keys: Tuple[str, ...] = ("text", "statement", "step")) -> List[Claim]:
        claims: List[Claim] = []
        if isinstance(raw_items, (str, dict)):
            raw_items = [raw_items]
        for raw in raw_items if isinstance(raw_items, list) else []:
            if isinstance(raw, dict):
                text = coerce_str(raw, text_keys)
                claims.append(self.validate(text, raw.get("type"), raw.get("evidence_ids") or raw.get("evidence")))
            else:
                text = coerce_str(raw)
                if text:
                    claims.append(self.validate(text, None, None))
        return [c for c in claims if c.text]

    # ------------------------------------------------------------------ #
    def _events(self, ids: Iterable[str]) -> List[LogEvent]:
        return [e for e in (self.lookup(i) for i in ids) if e is not None]

    def _has_confirmation(self, ids: List[str]) -> bool:
        return any("DIAGNOSIS" in e.tags or _CAUSED_BY_RE.search(e.raw) for e in self._events(ids))

    def _cites_text(self, ids: List[str], pattern: re.Pattern) -> bool:
        return any(pattern.search(e.raw) for e in self._events(ids))


def cap_confidence(value: float, claim_type: ClaimType) -> float:
    return round(min(value, CONFIDENCE_CAPS[claim_type]), 2)


def dict_get(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None
