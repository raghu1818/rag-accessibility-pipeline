"""Domain exceptions — never leak internal details to API consumers."""
from __future__ import annotations


class RAGPipelineError(Exception):
    """Base class for all pipeline errors."""


class DocumentIngestionError(RAGPipelineError):
    """Raised when a document cannot be processed."""


class OCRFallbackError(DocumentIngestionError):
    """Raised when both direct extraction and OCR fail."""


class VectorStoreError(RAGPipelineError):
    """Raised when the FAISS index operation fails."""


class RetrievalError(RAGPipelineError):
    """Raised when no relevant documents are found."""


class GenerationError(RAGPipelineError):
    """Raised when the LLM generation step fails."""


class HallucinationGuardError(GenerationError):
    """Raised when the answer cannot be grounded in retrieved context."""
