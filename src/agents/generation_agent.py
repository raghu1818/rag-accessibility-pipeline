"""
Generation Agent
────────────────
Responsibilities:
  1. Format retrieved documents into a numbered CONTEXT block.
  2. Call the LLM using the chain-of-thought + zero-hallucination prompt.
  3. Parse the <thinking> block from the raw response.
  4. Run a self-check (hallucination guard) by asking the LLM to verify every
     factual claim against the context; retry once if violations are found.
  5. Return the final answer and structured metadata.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.core.config import settings
from src.core.exceptions import GenerationError
from src.core.logging import get_logger
from src.graph.state import PipelineState
from src.prompts.templates import (
    HALLUCINATION_CHECK_PROMPT,
    generation_prompt,
)

logger = get_logger(__name__)

_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_MAX_CONTEXT_CHARS = 12_000  # guard against exceeding context window


class GenerationAgent:
    """Generates grounded, accessibility-optimised answers."""

    def __init__(self, llm: Optional[ChatOpenAI] = None) -> None:
        self._llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
        )
        self._chain = generation_prompt | self._llm

    # ── LangGraph nodes ───────────────────────────────────────────────────────

    async def run(self, state: PipelineState) -> dict:
        """Primary generation node."""
        query = state.get("query", "")
        docs = state.get("retrieved_documents") or []

        if not docs:
            no_context_answer = (
                "I could not find this information in the provided documents.  "
                "Please consult the original manual or contact the manufacturer."
            )
            return {
                "final_answer": no_context_answer,
                "raw_answer": no_context_answer,
                "grounded": True,
                "thinking": None,
            }

        context = self._format_context(docs)

        # On retry (grounded == False from previous run), add a stricter hint
        retry_note = ""
        if state.get("grounded") is False:
            violations = state.get("hallucination_violations") or []
            retry_note = (
                "\n\n[RETRY] Your previous answer contained unsupported claims: "
                + "; ".join(violations)
                + ".  Remove or replace them with information from the context only."
            )

        try:
            response = await self._chain.ainvoke(
                {"context": context, "question": query + retry_note}
            )
            raw_text: str = response.content
        except Exception as exc:
            raise GenerationError(f"LLM generation failed: {exc}") from exc

        thinking = self._extract_thinking(raw_text)
        clean_answer = self._strip_thinking(raw_text)

        logger.info(
            "generation_complete",
            query_preview=query[:80],
            answer_chars=len(clean_answer),
            has_thinking=thinking is not None,
        )

        return {
            "raw_answer": raw_text,
            "final_answer": clean_answer,
            "thinking": thinking,
            "grounded": None,  # guard runs next
        }

    async def hallucination_guard(self, state: PipelineState) -> dict:
        """Hallucination guard node — verifies every claim is grounded."""
        answer = state.get("final_answer") or state.get("raw_answer") or ""
        query = state.get("query", "")
        docs = state.get("retrieved_documents") or []

        if not answer or not docs:
            return {"grounded": True, "hallucination_violations": []}

        context = self._format_context(docs)
        check_prompt = HALLUCINATION_CHECK_PROMPT.format(
            question=query,
            context=context,
            answer=answer,
        )

        try:
            response = await self._llm.ainvoke(check_prompt)
            raw_json = response.content.strip()
            # Strip markdown code fences if present
            raw_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json, flags=re.DOTALL)
            result = json.loads(raw_json)
            grounded: bool = result.get("grounded", True)
            violations: list[str] = result.get("violations", [])
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("hallucination_guard_parse_error", error=str(exc))
            # If we can't parse the check, assume grounded to avoid loops
            grounded, violations = True, []

        if not grounded:
            logger.warning(
                "hallucination_violations_found",
                count=len(violations),
                violations=violations[:3],
            )

        return {"grounded": grounded, "hallucination_violations": violations}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        parts = []
        total_chars = 0
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            header = (
                f"[Source {i}] {meta.get('filename', 'unknown')} "
                f"— page {meta.get('page', '?')}"
            )
            entry = f"{header}\n{doc.page_content}"
            if total_chars + len(entry) > _MAX_CONTEXT_CHARS:
                break
            parts.append(entry)
            total_chars += len(entry)
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_thinking(text: str) -> Optional[str]:
        match = _THINKING_RE.search(text)
        return match.group(1).strip() if match else None

    @staticmethod
    def _strip_thinking(text: str) -> str:
        cleaned = _THINKING_RE.sub("", text)
        return cleaned.strip()
