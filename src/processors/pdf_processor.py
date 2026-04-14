"""
PDF text extractor with automatic OCR fallback.

Strategy
--------
1. Try native text extraction via PyMuPDF (fast, preserves layout well).
2. If a page yields fewer than MIN_CHARS characters (likely a scanned image),
   fall back to OCR via pytesseract.
3. After extraction, chunk the text using LangChain's RecursiveCharacterTextSplitter
   and attach rich metadata so the retrieval agent can surface source + page
   information to users.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.core.config import settings
from src.core.exceptions import DocumentIngestionError
from src.core.logging import get_logger
from src.processors.ocr_processor import OCRProcessor

logger = get_logger(__name__)

# Pages with fewer characters than this are considered image-based.
MIN_CHARS_NATIVE = 50


@dataclass
class PageExtractionResult:
    page_number: int
    text: str
    extraction_method: str  # "native" | "ocr"
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())


class PDFProcessor:
    """Processes a PDF file into chunked LangChain Documents."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._ocr = OCRProcessor()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, pdf_path: Path) -> list[Document]:
        """Extract, chunk and return Documents from *pdf_path*."""
        if not pdf_path.exists():
            raise DocumentIngestionError(f"File not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise DocumentIngestionError(
                f"Expected a .pdf file, got: {pdf_path.suffix}"
            )

        source_id = self._source_id(pdf_path)
        logger.info("processing_pdf", path=str(pdf_path), source_id=source_id)

        try:
            pages = self._extract_pages(pdf_path)
        except Exception as exc:
            raise DocumentIngestionError(
                f"Failed to open PDF '{pdf_path.name}': {exc}"
            ) from exc

        documents: list[Document] = []
        for page in pages:
            if not page.text.strip():
                continue
            chunks = self._splitter.split_text(page.text)
            for chunk_idx, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": str(pdf_path),
                            "source_id": source_id,
                            "filename": pdf_path.name,
                            "page": page.page_number,
                            "chunk": chunk_idx,
                            "extraction_method": page.extraction_method,
                            "word_count": len(chunk.split()),
                        },
                    )
                )

        logger.info(
            "pdf_processed",
            filename=pdf_path.name,
            pages=len(pages),
            chunks=len(documents),
        )
        return documents

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_pages(self, pdf_path: Path) -> list[PageExtractionResult]:
        results: list[PageExtractionResult] = []
        with fitz.open(str(pdf_path)) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                native_text = page.get_text("text").strip()

                if len(native_text) >= MIN_CHARS_NATIVE:
                    results.append(
                        PageExtractionResult(
                            page_number=page_num + 1,
                            text=native_text,
                            extraction_method="native",
                        )
                    )
                    logger.debug(
                        "page_extracted_native",
                        page=page_num + 1,
                        chars=len(native_text),
                    )
                else:
                    # Render page as image and apply OCR
                    ocr_text = self._ocr_page(doc, page, page_num)
                    results.append(
                        PageExtractionResult(
                            page_number=page_num + 1,
                            text=ocr_text,
                            extraction_method="ocr",
                        )
                    )
        return results

    def _ocr_page(
        self, doc: fitz.Document, page: fitz.Page, page_num: int
    ) -> str:
        logger.info("ocr_fallback_triggered", page=page_num + 1)
        try:
            # Render at configured DPI for best OCR accuracy
            mat = fitz.Matrix(settings.ocr_dpi / 72, settings.ocr_dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_bytes = pix.tobytes("png")
            return self._ocr.extract_text_from_bytes(img_bytes)
        except Exception as exc:
            logger.warning(
                "ocr_failed", page=page_num + 1, error=str(exc)
            )
            return ""

    @staticmethod
    def _source_id(path: Path) -> str:
        """Stable content-independent ID based on absolute path."""
        return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
