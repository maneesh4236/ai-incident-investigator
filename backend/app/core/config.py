"""
Centralized application configuration.

All settings are loaded from environment variables (or a `.env` file) so the
service can be reconfigured per-environment (local, docker-compose, CI, prod)
without touching code.
"""
from functools import lru_cache
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- General ---
    APP_NAME: str = "AetherLog-Inspired AI Incident Investigator"
    ENV: str = "development"
    API_V1_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # --- LLM (Gemini) ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIM: int = 1024

    # --- Qdrant ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "incident_chunks"
    QDRANT_MODE: str = "local"  # local (embedded, on disk) | server (QDRANT_HOST/PORT) | memory
    QDRANT_LOCAL_PATH: str = "./qdrant_data"
    QDRANT_POINT_MAX_CHARS: int = 700  # keeps each point inside the embedder's 256 word-piece window
    QDRANT_UPSERT_BATCH: int = 256

    # --- Neo4j ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    # --- Storage ---
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_MB: int = 50

    # --- Chunking (non-log documents only; logs are processed as events) ---
    CHUNK_SIZE_TOKENS: int = 400
    CHUNK_OVERLAP_TOKENS: int = 60

    # --- Retrieval ---
    TOP_K_VECTOR: int = 8
    TOP_K_GRAPH_HOPS: int = 2

    # --- Gemini call behaviour ---
    GEMINI_TIMEOUT_SECONDS: float = 60.0
    GEMINI_MAX_ATTEMPTS: int = 3
    GEMINI_BACKOFF_BASE_SECONDS: float = 1.0
    GEMINI_BACKOFF_MAX_SECONDS: float = 8.0
    GEMINI_MAX_TOTAL_SECONDS: float = 90.0
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_OUTPUT_TOKENS_INVESTIGATION: int = 2048
    GEMINI_MAX_OUTPUT_TOKENS_CHAT: int = 1024

    # --- Token budgets: hard limits on what is sent to Gemini ---
    MAX_GEMINI_CONTEXT_TOKENS: int = 6000  # whole prompt: system + instructions + evidence
    MAX_GEMINI_EVIDENCE_TOKENS: int = 4000  # evidence block of the investigation call
    MAX_CHAT_EVIDENCE_TOKENS: int = 3000
    MAX_CHAT_HISTORY_TOKENS: int = 600
    MAX_GRAPH_CONTEXT_TOKENS: int = 400
    TOKEN_ESTIMATE_CHARS_PER_TOKEN: float = 3.0  # conservative for log text (~3.4 chars/word-piece measured)

    # --- Event processing ---
    MAX_STACK_FRAMES_KEPT: int = 8
    PRECURSOR_WINDOW_SECONDS: int = 600
    TIMELINE_MAX_EVENTS: int = 30
    # Distinct occurrences of one template that may be shown individually
    # before the rest are summarised (first / peak / last always survive).
    MAX_INSTANCES_PER_GROUP: int = 6

    # --- Startup ---
    EMBEDDER_WARMUP_ON_STARTUP: bool = True

    @model_validator(mode="after")
    def _check_budgets(self) -> "Settings":
        if self.MAX_GEMINI_EVIDENCE_TOKENS >= self.MAX_GEMINI_CONTEXT_TOKENS:
            raise ValueError("MAX_GEMINI_EVIDENCE_TOKENS must be smaller than MAX_GEMINI_CONTEXT_TOKENS")
        if self.MAX_CHAT_EVIDENCE_TOKENS >= self.MAX_GEMINI_CONTEXT_TOKENS:
            raise ValueError("MAX_CHAT_EVIDENCE_TOKENS must be smaller than MAX_GEMINI_CONTEXT_TOKENS")
        if self.TOKEN_ESTIMATE_CHARS_PER_TOKEN <= 0:
            raise ValueError("TOKEN_ESTIMATE_CHARS_PER_TOKEN must be positive")
        if self.GEMINI_MAX_ATTEMPTS < 1:
            raise ValueError("GEMINI_MAX_ATTEMPTS must be >= 1")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
