"""
Health check endpoints.

GET /health       — liveness probe (always 200 if the process is running)
GET /health/ready — readiness probe (200 only when vector store is loaded)
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    environment: str
    document_count: int


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def liveness(request: Request) -> HealthResponse:
    pipeline = request.app.state.pipeline
    return HealthResponse(
        status="ok",
        environment=request.app.state.settings.environment,
        document_count=pipeline.retrieval_agent._store.document_count,
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness(request: Request) -> HealthResponse:
    pipeline = request.app.state.pipeline
    count = pipeline.retrieval_agent._store.document_count
    return HealthResponse(
        status="ready" if count >= 0 else "not_ready",
        environment=request.app.state.settings.environment,
        document_count=count,
    )
