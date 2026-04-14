"""
Retrieval Agent
───────────────
Responsibilities:
  1. Accept a natural-language query from the pipeline state.
  2. Run similarity search against the FAISS vector store.
  3. Apply a relevance score threshold to discard noise.
  4. Populate state.retrieved_documents for the generation agent.
  5. Log retrieval telemetry for observability.
"""

from __future__ import annotations

from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.core.logging import get_logger
from src.graph.state import PipelineState
from src.vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


class RetrievalAgent:
    """Fetches relevant document chunks from FAISS for a given query."""

    def __init__(self, vector_store: FAISSVectorStore | None = None) -> None:
        self._store = vector_store or FAISSVectorStore()

    # ── LangGraph node ────────────────────────────────────────────────────────

    async def run(self, state: PipelineState) -> dict:
        """LangGraph node entry point."""
        query = state.get("query")
        if not query:
            return {"error": "No query provided to retrieval agent."}

        threshold = state.get("retrieval_score_threshold") or settings.faiss_top_k
        score_threshold = float(threshold) if isinstance(threshold, (int, float)) else 0.3
        try:
            docs = self._retrieve(query, score_threshold=score_threshold)
        except RetrievalError as exc:
            logger.error("retrieval_failed", error=str(exc))
            return {"error": str(exc)}

        if not docs:
            logger.warning(
                "no_documents_retrieved",
                query_preview=query[:80],
                threshold=threshold,
            )

        return {"retrieved_documents": docs}

    # ── Public helpers ────────────────────────────────────────────────────────

    def retrieve(self, query: str, score_threshold: float = 0.3) -> list:
        return self._retrieve(query, score_threshold)

    # ── Private ───────────────────────────────────────────────────────────────

    def _retrieve(self, query: str, score_threshold: float) -> list:
        try:
            docs = self._store.similarity_search(
                query,
                k=settings.faiss_top_k,
                score_threshold=score_threshold,
            )
            logger.info(
                "retrieved",
                query_preview=query[:80],
                docs_returned=len(docs),
                threshold=score_threshold,
            )
            return docs
        except Exception as exc:
            raise RetrievalError(str(exc)) from exc
