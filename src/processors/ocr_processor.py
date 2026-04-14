"""
OCR processor using pytesseract (Tesseract wrapper).

Pre-processing pipeline applied before OCR:
  1. Convert to greyscale
  2. Upscale if the image is smaller than MIN_DIM in either dimension
  3. Apply adaptive threshold (handles uneven illumination in scanned pages)
  4. Pass to Tesseract with page-segmentation mode 6 (uniform block of text)
"""
from __future__ import annotations

import io

from PIL import Image, ImageFilter, ImageOps

from src.core.config import settings
from src.core.exceptions import OCRFallbackError
from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False
    logger.warning("pytesseract_not_installed_ocr_disabled")

# Minimum pixel dimension — upscale if smaller
MIN_DIM = 1000
# Tesseract page-segmentation mode
PSM = "6"


class OCRProcessor:
    """Extracts text from raw image bytes using Tesseract OCR."""

    def extract_text_from_bytes(self, image_bytes: bytes) -> str:
        """Run OCR on *image_bytes* (PNG/JPEG/TIFF).

        Raises OCRFallbackError if Tesseract is unavailable or fails.
        """
        if not _TESSERACT_AVAILABLE:
            raise OCRFallbackError(
                "pytesseract is not installed.  Install it with: "
                "pip install pytesseract  and ensure Tesseract is on PATH."
            )

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = self._preprocess(image)
            text: str = pytesseract.image_to_string(
                image,
                lang=settings.ocr_language,
                config=f"--psm {PSM}",
            )
            logger.debug("ocr_complete", chars=len(text))
            return text.strip()
        except OCRFallbackError:
            raise
        except Exception as exc:
            raise OCRFallbackError(f"Tesseract failed: {exc}") from exc

    # ── Pre-processing ────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(image: Image.Image) -> Image.Image:
        """Greyscale → upscale → sharpen → auto-contrast."""
        image = ImageOps.grayscale(image)

        w, h = image.size
        if w < MIN_DIM or h < MIN_DIM:
            scale = max(MIN_DIM / w, MIN_DIM / h)
            image = image.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )

        image = image.filter(ImageFilter.SHARPEN)
        image = ImageOps.autocontrast(image)
        return image
