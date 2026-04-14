"""
LangGraph shared state schema.

The state object is passed between every node in the graph.  All fields are
optional by default so that nodes may be entered at different stages
(e.g. retrieval-only or generation-only for unit testing).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class PipelineState(MessagesState):
    """State threaded through the multi-agent RAG graph."""

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # Set by the ingestion agent after a document is processed.
    ingested_source: str | None = None
    ingested_chunk_count: int | None = None

    # ── Retrieval ─────────────────────────────────────────────────────────────
    query: str | None = None
    retrieved_documents: list[Document] | None = None
    retrieval_score_threshold: float = 0.3

    # ── Generation ───────────────────────────────────────────────────────────
    raw_answer: str | None = None  # LLM output before validation
    final_answer: str | None = None  # Validated / cleaned answer
    thinking: str | None = None  # Extracted CoT reasoning
    grounded: bool | None = None  # Hallucination check result
    hallucination_violations: list[str] = []  # noqa: RUF012  # Unsupported claims (if any)

    # ── Routing ───────────────────────────────────────────────────────────────
    error: str | None = None
    next_node: str | None = None  # Explicit routing override
