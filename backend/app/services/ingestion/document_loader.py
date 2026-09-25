"""
Loads raw text out of uploaded documents (PDF, txt, markdown-ish reports).

Kept separate from `log_loader.py` because logs need line-oriented,
timestamp-aware parsing while documents need page-oriented text extraction.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from app.core.logging_config import get_logger
from app.models.schemas import DocumentType

logger = get_logger("ingestion.document_loader")


class DocumentLoader:
    """Extracts plain text from PDFs, runbooks, incident reports, and arch docs."""

    def load(self, path: str, doc_type: DocumentType) -> str:
        suffix = Path(path).suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(path)
        return self._load_text(path)

    @staticmethod
    def _load_pdf(path: str) -> str:
        logger.info(f"Extracting text from PDF: {path}")
        text_parts: list[str] = []
        with fitz.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(f"[page {page_number}]\n{page_text.strip()}")
        return "\n\n".join(text_parts)

    @staticmethod
    def _load_text(path: str) -> str:
        logger.info(f"Reading plain text file: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    @staticmethod
    def infer_doc_type(filename: str) -> DocumentType:
        name = filename.lower()
        if name.endswith(".log"):
            return DocumentType.LOG
        if name.endswith(".pdf"):
            return DocumentType.PDF
        if "runbook" in name:
            return DocumentType.RUNBOOK
        if "incident" in name or "postmortem" in name:
            return DocumentType.INCIDENT_REPORT
        if "arch" in name or "design" in name:
            return DocumentType.ARCHITECTURE_DOC
        return DocumentType.TEXT
