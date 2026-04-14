"""
FastAPI application factory.

Usage
─────
  Development:  uvicorn src.api.main:app --reload
  Production:   gunicorn src.api.main:app -k uvicorn.workers.UvicornWorker
  Docker:       CMD defined in Dockerfile
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import health_router, ingest_router, query_router
from src.core.config import settings
from src.core.logging import configure_logging, get_logger
from src.graph.pipeline import build_pipeline

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level, settings.environment)
    logger.info(
        "startup",
        environment=settings.environment,
        model=settings.llm_model,
    )

    pipeline = build_pipeline()
    app.state.pipeline = pipeline
    app.state.settings = settings

    logger.info("pipeline_ready")
    yield

    logger.info("shutdown")


def create_app() -> FastAPI:
    application = FastAPI(
        title="AccessBot RAG API",
        description=(
            "Production-grade Retrieval-Augmented Generation pipeline for the "
            "Accessibility Research Lab.  Helps blind and low-vision users "
            "navigate product manuals via natural language queries."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS (tighten in production via env) ──────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    application.include_router(health_router)
    application.include_router(ingest_router)
    application.include_router(query_router)

    # ── Global exception handler ──────────────────────────────────────────────
    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    return application


app = create_app()
