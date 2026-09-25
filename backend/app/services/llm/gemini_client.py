"""
Thin wrapper around Google's Gemini SDK (`google-genai`, the current
supported client — the older `google-generativeai` package is deprecated).

Centralizing all LLM calls here means every other service (entity extractor,
RCA analyzer, timeline builder, chat agent) talks to a single, testable
interface instead of importing the SDK directly. This also makes it trivial
to swap providers later (OpenAI, Claude, local model) by implementing the
same `generate` / `generate_json` interface.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger("llm.gemini_client")


class GeminiClient:
    def __init__(self):
        settings = get_settings()
        self._model_name = settings.GEMINI_MODEL
        self._configured = bool(settings.GEMINI_API_KEY)
        if self._configured:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        else:
            logger.warning(
                "GEMINI_API_KEY not set. GeminiClient will run in offline/stub mode "
                "and return heuristic fallbacks instead of real completions."
            )
            self._client = None

    @property
    def is_configured(self) -> bool:
        return self._configured

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        if not self._configured:
            return ""

        config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None
        response = self._client.models.generate_content(
            model=self._model_name, contents=prompt, config=config
        )
        return response.text or ""

    def generate_json(self, prompt: str, system_instruction: Optional[str] = None) -> Dict[str, Any]:
        """Calls the model and parses a JSON object out of the response.

        Falls back to an empty dict (never raises) so callers can layer
        heuristic fallbacks on top when the LLM is unavailable or returns
        malformed JSON — important for a hackathon demo that must keep
        working without live API keys.
        """
        raw = self.generate(prompt, system_instruction=system_instruction)
        if not raw:
            return {}
        cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from Gemini response even after regex fallback")
            return {}
