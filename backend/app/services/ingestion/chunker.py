"""
Splits document/log text into overlapping chunks sized for embedding.

Uses a simple whitespace-token approximation (no external tokenizer
dependency) which is accurate enough for chunk-sizing purposes while keeping
the ingestion pipeline dependency-light.
"""
from __future__ import annotations

import uuid
from typing import List

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import Chunk

logger = get_logger("ingestion.chunker")


class Chunker:
    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
        self.overlap = overlap or settings.CHUNK_OVERLAP_TOKENS

    def split(self, text: str, document_id: str, investigation_id: str) -> List[Chunk]:
        words = text.split()
        if not words:
            return []

        step = max(self.chunk_size - self.overlap, 1)
        chunks: List[Chunk] = []
        index = 0

        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if not window:
                continue
            chunk_text = " ".join(window)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    investigation_id=investigation_id,
                    text=chunk_text,
                    chunk_index=index,
                    metadata={"word_start": start, "word_end": start + len(window)},
                )
            )
            index += 1
            if start + self.chunk_size >= len(words):
                break

        logger.info(f"Chunked document {document_id} into {len(chunks)} chunks")
        return chunks
