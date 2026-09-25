"""
Produces short summaries of chunks/documents, used to keep prompts small
when many chunks are fed into downstream RCA / timeline reasoning steps.
"""
from __future__ import annotations

from typing import List

from app.core.logging_config import get_logger
from app.models.schemas import Chunk
from app.services.llm.gemini_client import GeminiClient

logger = get_logger("ingestion.summarizer")

_SYSTEM_PROMPT = (
    "You are an SRE assistant that writes extremely concise, factual summaries "
    "of log/incident text. Never speculate beyond what is stated."
)


class Summarizer:
    def __init__(self, llm_client: GeminiClient | None = None):
        self.llm_client = llm_client or GeminiClient()

    def summarize_chunk(self, chunk: Chunk) -> str:
        if not self.llm_client.is_configured:
            return self._heuristic_summary(chunk.text)

        prompt = (
            "Summarize the following log/incident excerpt in 1-2 sentences. "
            "Focus on errors, failures, and service names.\n\n"
            f"---\n{chunk.text}\n---"
        )
        summary = self.llm_client.generate(prompt, system_instruction=_SYSTEM_PROMPT)
        return summary.strip() or self._heuristic_summary(chunk.text)

    def summarize_many(self, chunks: List[Chunk]) -> List[str]:
        return [self.summarize_chunk(c) for c in chunks]

    @staticmethod
    def _heuristic_summary(text: str) -> str:
        """Offline fallback: just return the first sentence-ish fragment."""
        snippet = text.strip().replace("\n", " ")
        return (snippet[:180] + "...") if len(snippet) > 180 else snippet
