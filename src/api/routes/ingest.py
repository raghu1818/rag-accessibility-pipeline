"""
Document ingestion endpoint.

POST /ingest — accepts a multipart PDF upload, processes it, and indexes it.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from src.api.schemas.ingest import IngestResponse
from src.core.exceptions import DocumentIngestionError
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a PDF document",
    description=(
        "Upload a PDF file for text extraction and vector indexing.  "
        "Scanned pages are automatically processed with OCR."
    ),
)
async def ingest_document(
    request: Request,
    file: UploadFile,
) -> IngestResponse:
    app_settings = request.app.state.settings
    pipeline = request.app.state.pipeline

    # ── Validation ────────────────────────────────────────────────────────────
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted.  Got: {file.content_type}",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a .pdf extension.",
        )

    # ── Persist upload ────────────────────────────────────────────────────────
    safe_stem = Path(file.filename).stem[:64]  # limit length
    dest_path = app_settings.upload_dir / f"{safe_stem}_{uuid.uuid4().hex[:8]}.pdf"

    size = 0
    try:
        with dest_path.open("wb") as out:
            while chunk := await file.read(65536):
                size += len(chunk)
                if size > app_settings.max_upload_bytes:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File exceeds maximum size of "
                            f"{app_settings.max_upload_size_mb} MB."
                        ),
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        logger.error("upload_write_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc

    logger.info(
        "file_uploaded",
        filename=file.filename,
        size_bytes=size,
        dest=str(dest_path),
    )

    # ── Ingest ────────────────────────────────────────────────────────────────
    try:
        result = await pipeline.ingest(str(dest_path))
    except DocumentIngestionError as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        logger.exception("unexpected_ingestion_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during ingestion.",
        ) from exc

    return IngestResponse(
        message=f"Successfully indexed '{file.filename}'.",
        **result,
        error=None,
    )
