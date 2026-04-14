"""
Application configuration via environment variables.
All secrets must be set in the environment or a .env file — never hardcoded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    llm_model: str = Field("gpt-4o", description="Chat model identifier")
    embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model identifier"
    )
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(2048, gt=0)

    # ── Vector store ─────────────────────────────────────────────────────────
    faiss_index_path: Path = Field(
        Path("data/processed/faiss_index"),
        description="Directory where the FAISS index is persisted",
    )
    faiss_top_k: int = Field(5, ge=1, le=50)
    chunk_size: int = Field(512, gt=0)
    chunk_overlap: int = Field(64, ge=0)

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000, gt=0)
    api_workers: int = Field(1, gt=0)
    api_reload: bool = Field(False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    environment: Literal["development", "staging", "production"] = "production"

    # ── OCR ───────────────────────────────────────────────────────────────────
    tesseract_cmd: str = Field(
        "tesseract", description="Path or command for Tesseract binary"
    )
    ocr_language: str = Field("eng", description="Tesseract language code(s)")
    ocr_dpi: int = Field(300, gt=0)

    # ── Upload ────────────────────────────────────────────────────────────────
    upload_dir: Path = Field(Path("data/raw"))
    max_upload_size_mb: int = Field(50, gt=0)

    @field_validator("faiss_index_path", "upload_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()  # type: ignore[call-arg]
