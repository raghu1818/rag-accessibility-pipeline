from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    message: str = Field(..., description="Human-readable status message")
    source: str = Field(..., description="Absolute path of the ingested file")
    filename: str = Field(..., description="Original filename")
    chunk_count: int = Field(..., description="Number of text chunks indexed")
    pages: int = Field(..., description="Total pages processed")
    extraction_methods: list[str] = Field(
        ...,
        description="Extraction methods used: 'native' and/or 'ocr'",
    )
    error: str | None = Field(None)
