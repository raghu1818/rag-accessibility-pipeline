"""
Thread-safe FAISS vector store wrapper.

Provides:
  • add_documents  — embed and index LangChain Document objects
  • similarity_search — top-k nearest-neighbour retrieval with score threshold
  • persist / load — serialise the FAISS index + docstore to disk
  • delete_by_source — remove all chunks belonging to a given source file
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.core.config import settings
from src.core.exceptions import VectorStoreError
from src.core.logging import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """Singleton-style wrapper around LangChain's FAISS integration."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        top_k: int | None = None,
    ) -> None:
        self._index_path = index_path or settings.faiss_index_path
        self._top_k = top_k or settings.faiss_top_k
        self._lock = threading.Lock()
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._store: Optional[FAISS] = None
        self._try_load()

    # ── Public interface ──────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed *documents* and add them to the index.  Returns inserted IDs."""
        if not documents:
            return []
        try:
            with self._lock:
                if self._store is None:
                    self._store = FAISS.from_documents(
                        documents, self._embeddings
                    )
                    ids = list(self._store.index_to_docstore_id.values())
                else:
                    ids = self._store.add_documents(documents)
                self._persist()
            logger.info(
                "indexed_documents", count=len(documents), ids_sample=ids[:3]
            )
            return ids
        except Exception as exc:
            raise VectorStoreError(f"Failed to index documents: {exc}") from exc

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        """Return at most *k* documents most similar to *query*."""
        if self._store is None:
            logger.warning("similarity_search_called_on_empty_store")
            return []
        k = k or self._top_k
        try:
            with self._lock:
                results = self._store.similarity_search_with_relevance_scores(
                    query, k=k
                )
            docs = [
                doc
                for doc, score in results
                if score >= score_threshold
            ]
            logger.info(
                "retrieval_complete",
                query_preview=query[:80],
                returned=len(docs),
                requested=k,
            )
            return docs
        except Exception as exc:
            raise VectorStoreError(f"Retrieval failed: {exc}") from exc

    def delete_by_source(self, source: str) -> int:
        """Remove all chunks whose metadata['source'] == *source*.

        Returns the number of chunks deleted.
        """
        if self._store is None:
            return 0
        with self._lock:
            ids_to_delete = [
                doc_id
                for doc_id, doc in self._store.docstore._dict.items()
                if doc.metadata.get("source") == source
            ]
            if ids_to_delete:
                self._store.delete(ids_to_delete)
                self._persist()
        logger.info("deleted_chunks", source=source, count=len(ids_to_delete))
        return len(ids_to_delete)

    @property
    def document_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

    # ── Private helpers ───────────────────────────────────────────────────────

    def _persist(self) -> None:
        """Serialise the in-memory index to disk (called under lock)."""
        if self._store is None:
            return
        self._store.save_local(str(self._index_path))
        logger.debug("faiss_index_persisted", path=str(self._index_path))

    def _try_load(self) -> None:
        """Attempt to load a previously serialised index from disk."""
        index_file = self._index_path / "index.faiss"
        if not index_file.exists():
            logger.info(
                "faiss_index_not_found_starting_fresh",
                path=str(self._index_path),
            )
            return
        try:
            self._store = FAISS.load_local(
                str(self._index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "faiss_index_loaded",
                path=str(self._index_path),
                documents=self._store.index.ntotal,
            )
        except Exception as exc:
            logger.warning("faiss_index_load_failed", error=str(exc))
            self._store = None
