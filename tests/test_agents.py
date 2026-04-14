"""Tests for the three pipeline agents."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.agents.generation_agent import GenerationAgent
from src.agents.ingestion_agent import IngestionAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.graph.state import PipelineState


# ── Ingestion Agent ───────────────────────────────────────────────────────────

class TestIngestionAgent:
    def test_ingest_file_returns_summary(
        self, test_pdf: Path, mock_vector_store: MagicMock
    ) -> None:
        agent = IngestionAgent(vector_store=mock_vector_store)
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            agent.ingest_file(str(test_pdf))
        )
        assert result["chunk_count"] > 0
        assert result["filename"] == test_pdf.name
        mock_vector_store.add_documents.assert_called_once()

    def test_deduplication_deletes_stale_chunks(
        self, test_pdf: Path, mock_vector_store: MagicMock
    ) -> None:
        mock_vector_store.delete_by_source.return_value = 5
        agent = IngestionAgent(vector_store=mock_vector_store)
        import asyncio

        asyncio.get_event_loop().run_until_complete(agent.ingest_file(str(test_pdf)))
        mock_vector_store.delete_by_source.assert_called_once_with(str(test_pdf))


# ── Retrieval Agent ───────────────────────────────────────────────────────────

class TestRetrievalAgent:
    def test_retrieves_documents(
        self, mock_vector_store: MagicMock, sample_documents: list[Document]
    ) -> None:
        agent = RetrievalAgent(vector_store=mock_vector_store)
        docs = agent.retrieve("How do I enable accessibility mode?")
        assert docs == sample_documents
        mock_vector_store.similarity_search.assert_called_once()

    def test_empty_query_returns_error_in_state(
        self, mock_vector_store: MagicMock
    ) -> None:
        import asyncio

        agent = RetrievalAgent(vector_store=mock_vector_store)
        state = PipelineState(messages=[], query=None)
        result = asyncio.get_event_loop().run_until_complete(agent.run(state))
        assert "error" in result


# ── Generation Agent ──────────────────────────────────────────────────────────

class TestGenerationAgent:
    def test_generates_answer(
        self,
        mock_llm: MagicMock,
        sample_documents: list[Document],
    ) -> None:
        import asyncio

        agent = GenerationAgent(llm=mock_llm)
        state = PipelineState(
            messages=[],
            query="How do I enable accessibility mode?",
            retrieved_documents=sample_documents,
        )
        result = asyncio.get_event_loop().run_until_complete(agent.run(state))
        assert result["final_answer"]
        assert "accessibility mode" in result["final_answer"].lower()

    def test_extracts_thinking_block(
        self,
        mock_llm: MagicMock,
        sample_documents: list[Document],
    ) -> None:
        import asyncio

        agent = GenerationAgent(llm=mock_llm)
        state = PipelineState(
            messages=[],
            query="test",
            retrieved_documents=sample_documents,
        )
        result = asyncio.get_event_loop().run_until_complete(agent.run(state))
        assert result["thinking"] is not None
        assert "thinking" in result["thinking"].lower() or len(result["thinking"]) > 0

    def test_no_documents_returns_fallback(self, mock_llm: MagicMock) -> None:
        import asyncio

        agent = GenerationAgent(llm=mock_llm)
        state = PipelineState(
            messages=[],
            query="test",
            retrieved_documents=[],
        )
        result = asyncio.get_event_loop().run_until_complete(agent.run(state))
        assert "could not find" in result["final_answer"].lower()
        assert result["grounded"] is True

    def test_hallucination_guard_grounded(
        self,
        mock_llm: MagicMock,
        sample_documents: list[Document],
    ) -> None:
        import asyncio

        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content='{"grounded": true, "violations": []}')
        )
        agent = GenerationAgent(llm=mock_llm)
        state = PipelineState(
            messages=[],
            query="test",
            retrieved_documents=sample_documents,
            final_answer="Press and hold the side button. [Source 1]",
        )
        result = asyncio.get_event_loop().run_until_complete(
            agent.hallucination_guard(state)
        )
        assert result["grounded"] is True
        assert result["hallucination_violations"] == []
