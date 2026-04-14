"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient


class TestHealthEndpoints:
    def test_liveness(self, api_client: TestClient) -> None:
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "document_count" in data

    def test_readiness(self, api_client: TestClient) -> None:
        resp = api_client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("ready", "not_ready")


class TestQueryEndpoint:
    def test_valid_query_returns_answer(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/query",
            json={"question": "How do I enable accessibility mode?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["sources"], list)

    def test_thinking_hidden_by_default(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/query",
            json={"question": "How do I change the language?"},
        )
        assert resp.json().get("thinking") is None

    def test_thinking_included_when_requested(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/query",
            json={
                "question": "How do I change the language?",
                "include_thinking": True,
            },
        )
        assert resp.status_code == 200
        # thinking may be None if the LLM mock doesn't include <thinking> tags
        # but the field should be present in the response
        assert "thinking" in resp.json()

    def test_short_question_rejected(self, api_client: TestClient) -> None:
        resp = api_client.post("/query", json={"question": "hi"})
        assert resp.status_code == 422

    def test_missing_question_rejected(self, api_client: TestClient) -> None:
        resp = api_client.post("/query", json={})
        assert resp.status_code == 422


class TestIngestEndpoint:
    def test_non_pdf_rejected(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/ingest",
            files={
                "file": (
                    "document.txt",
                    io.BytesIO(b"hello"),
                    "text/plain",
                )
            },
        )
        assert resp.status_code == 415

    def test_pdf_upload_accepted(self, api_client: TestClient, test_pdf: Path) -> None:
        with test_pdf.open("rb") as f:
            resp = api_client.post(
                "/ingest",
                files={"file": ("manual.pdf", f, "application/pdf")},
            )
        assert resp.status_code == 202
        data = resp.json()
        assert "chunk_count" in data
        assert data["chunk_count"] > 0
