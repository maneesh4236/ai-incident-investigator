"""
FastAPI application entrypoint for the AI Incident Investigator backend.
"""
from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, graph, investigate, upload
from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger
from app.core.metrics import process_totals

settings = get_settings()
configure_logging()
logger = get_logger("main")

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered Root Cause Analysis and Incident Investigation Platform.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix=settings.API_V1_PREFIX)
app.include_router(investigate.router, prefix=settings.API_V1_PREFIX)
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)
app.include_router(graph.router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}


@app.get(f"{settings.API_V1_PREFIX}/metrics")
async def metrics():
    """Process-wide totals (Gemini calls/attempts/errors/tokens, events, Qdrant/graph ops)."""
    return process_totals()


@app.on_event("startup")
async def on_startup():
    logger.info(f"{settings.APP_NAME} starting up (env={settings.ENV})")
    if settings.EMBEDDER_WARMUP_ON_STARTUP:
        # Load the embedding model in the background so the first upload does
        # not pay the cold-load cost (measured 13-43 s) on its request.
        from app.core.dependencies import get_embedder

        threading.Thread(target=get_embedder().warmup, name="embedder-warmup", daemon=True).start()
