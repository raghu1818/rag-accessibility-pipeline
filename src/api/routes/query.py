"""
Query endpoint.

POST /query — accepts a natural-language question and returns a grounded answer.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas.query import QueryRequest, QueryResponse, SourceReference
from src.core.exceptions import GenerationError, RetrievalError
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the indexed documents",
    description=(
        "Ask a question about the ingested product manuals.  "
        "The response is grounded in the retrieved passages and includes "
        "source citations.  Set include_thinking=true to inspect reasoning."
    ),
)
async def query_documents(
    request: Request,
    body: QueryRequest,
) -> QueryResponse:
    pipeline = request.app.state.pipeline

    logger.info(
        "query_received",
        question_preview=body.question[:80],
        threshold=body.score_threshold,
    )

    try:
        result = await pipeline.query(
            question=body.question,
            score_threshold=body.score_threshold,
        )
    except (RetrievalError, GenerationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("unexpected_query_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your query.",
        )

    if result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    sources = [SourceReference(**s) for s in result.get("sources", [])]
    thinking = result.get("thinking") if body.include_thinking else None

    return QueryResponse(
        answer=result["answer"],
        grounded=result.get("grounded"),
        sources=sources,
        thinking=thinking,
        error=None,
    )
