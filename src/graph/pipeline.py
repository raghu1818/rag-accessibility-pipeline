"""
LangGraph pipeline wiring the three agents together.

Graph topology
──────────────
                ┌──────────────────┐
                │   ingestion_node │  (optional — only on /ingest calls)
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │  retrieval_node  │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │ generation_node  │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │  hallucination   │
                │  guard_node      │
                └────────┬─────────┘
                         │
                     END / retry (max 1)

All nodes share PipelineState.  Errors set state.error and route to END.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.generation_agent import GenerationAgent
from src.agents.ingestion_agent import IngestionAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.core.logging import get_logger
from src.graph.state import PipelineState

logger = get_logger(__name__)

_INGESTION = "ingestion"
_RETRIEVAL = "retrieval"
_GENERATION = "generation"
_GUARD = "hallucination_guard"


def _route_after_guard(state: PipelineState) -> str:
    """Re-generate once if grounding check fails; otherwise finish."""
    if state.grounded is False and not state.error:
        # Allow a single retry with a stricter prompt hint injected by the
        # generation agent when it detects state.grounded == False.
        if state.hallucination_violations:
            logger.warning(
                "hallucination_detected_retrying",
                violations=state.hallucination_violations,
            )
            return _GENERATION
    return END


def build_pipeline() -> "RAGPipeline":
    """Construct and compile the LangGraph state machine."""
    ingestion_agent = IngestionAgent()
    retrieval_agent = RetrievalAgent()
    generation_agent = GenerationAgent()

    graph = StateGraph(PipelineState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node(_INGESTION, ingestion_agent.run)
    graph.add_node(_RETRIEVAL, retrieval_agent.run)
    graph.add_node(_GENERATION, generation_agent.run)
    graph.add_node(_GUARD, generation_agent.hallucination_guard)

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.add_edge(_INGESTION, _RETRIEVAL)
    graph.add_edge(_RETRIEVAL, _GENERATION)
    graph.add_edge(_GENERATION, _GUARD)
    graph.add_conditional_edges(_GUARD, _route_after_guard)

    # Entry point — retrieval by default; callers can override via state
    graph.set_entry_point(_RETRIEVAL)

    compiled = graph.compile()
    logger.info("langgraph_pipeline_compiled")
    return RAGPipeline(
        compiled=compiled,
        ingestion_agent=ingestion_agent,
        retrieval_agent=retrieval_agent,
        generation_agent=generation_agent,
    )


class RAGPipeline:
    """High-level wrapper around the compiled LangGraph app."""

    def __init__(
        self,
        compiled: Any,
        ingestion_agent: IngestionAgent,
        retrieval_agent: RetrievalAgent,
        generation_agent: GenerationAgent,
    ) -> None:
        self._app = compiled
        self.ingestion_agent = ingestion_agent
        self.retrieval_agent = retrieval_agent
        self.generation_agent = generation_agent

    async def query(self, question: str, score_threshold: float = 0.3) -> dict:
        """Run the retrieval → generation → guard path."""
        initial_state: dict = {
            "query": question,
            "retrieval_score_threshold": score_threshold,
            "messages": [],
        }
        result: PipelineState = await self._app.ainvoke(initial_state)
        return {
            "answer": result.get("final_answer") or result.get("raw_answer", ""),
            "thinking": result.get("thinking"),
            "grounded": result.get("grounded"),
            "sources": _format_sources(result.get("retrieved_documents") or []),
            "error": result.get("error"),
        }

    async def ingest(self, pdf_path: str) -> dict:
        """Run the ingestion → retrieval (skipped) → END path."""
        result = await self.ingestion_agent.ingest_file(pdf_path)
        return result


def _format_sources(docs: list) -> list[dict]:
    seen: set[tuple] = set()
    sources = []
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "filename": doc.metadata.get("filename", "unknown"),
                    "page": doc.metadata.get("page"),
                    "extraction_method": doc.metadata.get("extraction_method"),
                }
            )
    return sources
