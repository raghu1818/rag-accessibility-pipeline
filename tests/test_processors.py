"""Tests for PDF text extraction and OCR processor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import DocumentIngestionError
from src.processors.pdf_processor import PDFProcessor


class TestPDFProcessor:
    def test_processes_native_pdf(self, test_pdf: Path) -> None:
        processor = PDFProcessor(chunk_size=200, chunk_overlap=20)
        docs = processor.process(test_pdf)
        assert len(docs) > 0
        assert all(d.page_content for d in docs)
        assert all(d.metadata["filename"] == test_pdf.name for d in docs)

    def test_metadata_populated(self, test_pdf: Path) -> None:
        processor = PDFProcessor(chunk_size=200, chunk_overlap=20)
        docs = processor.process(test_pdf)
        for doc in docs:
            assert "page" in doc.metadata
            assert "chunk" in doc.metadata
            assert "source_id" in doc.metadata
            assert doc.metadata["extraction_method"] in ("native", "ocr")

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        processor = PDFProcessor()
        with pytest.raises(DocumentIngestionError, match="File not found"):
            processor.process(tmp_path / "nonexistent.pdf")

    def test_raises_on_wrong_extension(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("hello")
        processor = PDFProcessor()
        with pytest.raises(DocumentIngestionError, match=r"Expected a \.pdf"):
            processor.process(txt_file)

    def test_ocr_fallback_triggered_for_image_pages(self, tmp_path: Path) -> None:
        """Pages with < MIN_CHARS_NATIVE native text should trigger OCR."""
        import fitz

        pdf_path = tmp_path / "blank.pdf"
        doc = fitz.open()
        doc.new_page()  # blank page — no native text
        doc.save(str(pdf_path))
        doc.close()

        ocr_mock = MagicMock()
        ocr_mock.extract_text_from_bytes.return_value = "OCR extracted text here."

        with patch("src.processors.pdf_processor.OCRProcessor", return_value=ocr_mock):
            processor = PDFProcessor(chunk_size=200, chunk_overlap=0)
            docs = processor.process(pdf_path)

        ocr_mock.extract_text_from_bytes.assert_called_once()
        assert any(d.metadata["extraction_method"] == "ocr" for d in docs)
