"""
ocr_processor.py
-----------------
Adds OCR (Optical Character Recognition) as a fallback for PDF pages
that have little/no extractable text — i.e. scanned documents or
image-only pages.

APPROACH:
1. `pdf_extractor.py` already flags each page as `is_likely_scanned`
   based on how much text PyMuPDF could pull out directly.
2. For flagged pages ONLY, we render that page to a high-resolution
   image (reusing `render_page_as_image` from pdf_extractor — no new
   dependency needed there, PyMuPDF does the rendering).
3. We run Tesseract (via `pytesseract`) on that image to recover text.

This "hybrid" approach (real text layer first, OCR only where needed)
is deliberately faster and more accurate than OCR'ing every page —
native PDF text is always more reliable than OCR when it's available.

SYSTEM DEPENDENCY:
Tesseract must be installed at the OS level (this is a separate binary,
not a Python package):
  - macOS:   brew install tesseract
  - Ubuntu:  sudo apt-get install tesseract-ocr
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
If `pytesseract.get_tesseract_version()` fails, `is_ocr_available()`
below returns False and the UI will show a clear message instead of
crashing.
"""

import io

from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

from src.document_processing.pdf_extractor import PDFExtractionResult, render_page_as_image
from src.utils.logger import logger

# Higher zoom = sharper image = better OCR accuracy, but slower.
# 2.5-3.0 is a good accuracy/speed tradeoff for typical scanned documents.
OCR_ZOOM = 2.5


def is_ocr_available() -> bool:
    """Checks whether the Tesseract binary is installed and reachable."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        logger.warning(f"Tesseract OCR not available: {e}")
        return False


def ocr_page(pdf_path: str, page_number: int, zoom: float = OCR_ZOOM) -> str:
    """
    Renders one page of a PDF to an image and runs OCR on it.
    `page_number` is 1-indexed to match PDFExtractionResult.pages.
    """
    png_bytes = render_page_as_image(pdf_path, page_number, zoom=zoom)
    image = Image.open(io.BytesIO(png_bytes))
    text = pytesseract.image_to_string(image)
    return text.strip()


def apply_ocr_fallback(pdf_path: str, extraction: PDFExtractionResult) -> PDFExtractionResult:
    """
    Mutates and returns `extraction` in place: for every page flagged
    as `is_likely_scanned`, replaces its (empty/near-empty) text with
    OCR output, and rebuilds `full_text` from the updated pages.

    If Tesseract isn't installed, logs a warning and returns the
    extraction unchanged — the app should still work, just without
    scanned-page recovery.
    """
    if not is_ocr_available():
        logger.warning("Skipping OCR fallback — Tesseract is not installed on this system.")
        return extraction

    scanned_pages = [p for p in extraction.pages if p.is_likely_scanned]
    if not scanned_pages:
        return extraction

    logger.info(f"Running OCR fallback on {len(scanned_pages)} likely-scanned page(s)...")

    for page in scanned_pages:
        try:
            ocr_text = ocr_page(pdf_path, page.page_number)
            if ocr_text:
                logger.info(f"  Page {page.page_number}: OCR recovered {len(ocr_text)} characters")
                page.text = ocr_text
                page.char_count = len(ocr_text)
                page.is_likely_scanned = False  # recovered — no longer "unreadable"
            else:
                logger.warning(f"  Page {page.page_number}: OCR found no text (image may be blank/unreadable).")
        except Exception as e:
            logger.error(f"  Page {page.page_number}: OCR failed — {e}")

    # Rebuild full_text now that some pages have new content
    extraction.full_text = "\n\n".join(p.text for p in extraction.pages)
    return extraction
