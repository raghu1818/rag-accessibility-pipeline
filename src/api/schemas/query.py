from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language question about the ingested documents",
        examples=["How do I enable the accessibility mode on this device?"],
    )
    score_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score (0-1) for retrieved passages",
    )
    include_thinking: bool = Field(
        False,
        description="Return the chain-of-thought reasoning block",
    )


class SourceReference(BaseModel):
    filename: str
    page: int | None = None
    extraction_method: str | None = None


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Grounded answer from AccessBot")
    grounded: bool | None = Field(
        None,
        description="True if all claims were verified against source passages",
    )
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Document passages used to generate the answer",
    )
    thinking: str | None = Field(
        None,
        description="Chain-of-thought reasoning (only when include_thinking=True)",
    )
    error: str | None = Field(None)
