"""
Ingestion Agent
───────────────
Responsibilities:
  1. Accept a PDF file path (from an API upload or a local path).
  2. Delegate text extraction to PDFProcessor (native + OCR fallback).
  3. Upsert the resulting Document chunks into the FAISS vector store,
     de-duplicating by source so re-uploading a file refreshes its content.
  4. Return structured metadata for the API response and LangGraph state.
"""

from __future__ import annotations

from pathlib import Path

from src.core.exceptions import DocumentIngestionError
from src.core.logging import get_logger
from src.graph.state import PipelineState
from src.processors.pdf_processor import PDFProcessor
from src.vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


class IngestionAgent:
    """Processes PDF files and populates the vector store."""

    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        pdf_processor: PDFProcessor | None = None,
    ) -> None:
        self._store = vector_store or FAISSVectorStore()
        self._processor = pdf_processor or PDFProcessor()

    # ── LangGraph node ────────────────────────────────────────────────────────

    async def run(self, state: PipelineState) -> dict:
        """LangGraph node entry point.  Reads source from state.messages[-1]."""
        try:
            # The API puts the file path in the last human message content
            messages = state.get("messages", [])
            if not messages:
                return {"error": "No file path provided to ingestion agent."}

            source_path = messages[-1].content
            result = await self.ingest_file(source_path)
            return {
                "ingested_source": result["source"],
                "ingested_chunk_count": result["chunk_count"],
            }
        except DocumentIngestionError as exc:
            logger.error("ingestion_failed", error=str(exc))
            return {"error": str(exc)}

    # ── Public helper (called directly by API + pipeline.ingest) ─────────────

    async def ingest_file(self, path: str) -> dict:
        """Process *path* and upsert into the vector store.

        Returns a summary dict suitable for API responses.
        """
        pdf_path = Path(path)
        logger.info("ingestion_started", path=str(pdf_path))

        # Remove stale chunks for this source before re-indexing
        deleted = self._store.delete_by_source(str(pdf_path))
        if deleted:
            logger.info("stale_chunks_removed", count=deleted, source=str(pdf_path))

        documents = self._processor.process(pdf_path)
        if not documents:
            raise DocumentIngestionError(
                f"No text could be extracted from '{pdf_path.name}'.  "
                "The file may be empty or in an unsupported format."
            )

        ids = self._store.add_documents(documents)

        result = {
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "chunk_count": len(ids),
            "pages": max((d.metadata.get("page", 0) for d in documents), default=0),
            "extraction_methods": list(
                {d.metadata.get("extraction_method", "unknown") for d in documents}
            ),
        }
        logger.info("ingestion_complete", **result)
        return result
