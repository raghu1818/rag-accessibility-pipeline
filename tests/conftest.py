"""
Pytest fixtures shared across the test suite.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

# ── Ensure env vars are set before importing app code ────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "sk-test-000000000000000000000000")
os.environ.setdefault("ENVIRONMENT", "development")


@pytest.fixture()
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "To enable accessibility mode, press and hold the side button "
                "for three seconds until you hear two beeps."
            ),
            metadata={
                "source": "/data/raw/manual.pdf",
                "source_id": "abc123",
                "filename": "manual.pdf",
                "page": 12,
                "chunk": 0,
                "extraction_method": "native",
            },
        ),
        Document(
            page_content=(
                "The device supports 18 languages for voice navigation.  "
                "Change the language in Settings > Accessibility > Language."
            ),
            metadata={
                "source": "/data/raw/manual.pdf",
                "source_id": "abc123",
                "filename": "manual.pdf",
                "page": 14,
                "chunk": 0,
                "extraction_method": "native",
            },
        ),
    ]


@pytest.fixture()
def mock_vector_store(sample_documents: list[Document]) -> MagicMock:
    store = MagicMock()
    store.similarity_search.return_value = sample_documents
    store.add_documents.return_value = ["id-1", "id-2"]
    store.delete_by_source.return_value = 0
    store.document_count = 2
    return store


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=(
                "<thinking>The context clearly states the procedure.</thinking>\n"
                "To enable accessibility mode, press and hold the side button "
                "for three seconds until you hear two beeps. [Source 1]"
            )
        )
    )
    return llm


@pytest.fixture()
def test_pdf(tmp_path: Path) -> Path:
    """Create a minimal valid PDF for ingestion tests."""
    import fitz

    pdf_path = tmp_path / "test_manual.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 100),
        "Accessibility Manual\n\nPress side button for accessibility mode.",
        fontsize=12,
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture()
def api_client(mock_vector_store: MagicMock, mock_llm: MagicMock) -> Generator:
    """TestClient with mocked LLM and vector store."""
    with (
        patch("src.vector_store.faiss_store.FAISSVectorStore.__init__", return_value=None),
        patch("src.vector_store.faiss_store.FAISSVectorStore._try_load"),
        patch("src.agents.retrieval_agent.FAISSVectorStore", return_value=mock_vector_store),
        patch("src.agents.ingestion_agent.FAISSVectorStore", return_value=mock_vector_store),
        patch("src.agents.generation_agent.ChatOpenAI", return_value=mock_llm),
    ):
        from src.api.main import create_app

        app = create_app()
        with TestClient(app) as client:
            yield client
