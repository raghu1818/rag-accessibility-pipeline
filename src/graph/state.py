"""
LangGraph shared state schema.

The state object is passed between every node in the graph.  All fields are
optional by default so that nodes may be entered at different stages
(e.g. retrieval-only or generation-only for unit testing).
"""
from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class PipelineState(MessagesState):
    """State threaded through the multi-agent RAG graph."""

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # Set by the ingestion agent after a document is processed.
    ingested_source: Optional[str] = None
    ingested_chunk_count: Optional[int] = None

    # ── Retrieval ─────────────────────────────────────────────────────────────
    query: Optional[str] = None
    retrieved_documents: Optional[list[Document]] = None
    retrieval_score_threshold: float = 0.3

    # ── Generation ───────────────────────────────────────────────────────────
    raw_answer: Optional[str] = None          # LLM output before validation
    final_answer: Optional[str] = None        # Validated / cleaned answer
    thinking: Optional[str] = None            # Extracted CoT reasoning
    grounded: Optional[bool] = None           # Hallucination check result
    hallucination_violations: list[str] = []  # Unsupported claims (if any)

    # ── Routing ───────────────────────────────────────────────────────────────
    error: Optional[str] = None
    next_node: Optional[str] = None           # Explicit routing override
