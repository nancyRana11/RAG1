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
   image (reusing `render_page_as_image` from pdf_extractor).
3. We run Tesseract (via `pytesseract`) on that image to recover text.

This "hybrid" approach (real text layer first, OCR only where needed)
is deliberately faster and more accurate than OCR'ing every page —
native PDF text is always more reliable than OCR when it's available.

SYSTEM DEPENDENCY:
Tesseract must be installed at the OS level.

Windows:
    C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""

import io
import os

from PIL import Image
import pytesseract

from src.document_processing.pdf_extractor import (
    PDFExtractionResult,
    render_page_as_image
)
from src.utils.logger import logger


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    raise RuntimeError(
        f"Tesseract not found at: {TESSERACT_PATH}"
    )


# ============================================================
# OCR SETTINGS
# ============================================================

# Higher zoom = sharper image = better OCR accuracy,
# but slower processing.
OCR_ZOOM = 2.5


# ============================================================
# CHECK OCR AVAILABILITY
# ============================================================

def is_ocr_available() -> bool:
    """
    Checks whether the Tesseract binary is installed and reachable.
    """
    try:
        pytesseract.get_tesseract_version()
        return True

    except Exception as e:
        logger.warning(f"Tesseract OCR not available: {e}")
        return False


# ============================================================
# OCR A SINGLE PAGE
# ============================================================

def ocr_page(
    pdf_path: str,
    page_number: int,
    zoom: float = OCR_ZOOM
) -> str:
    """
    Renders one page of a PDF to an image and runs OCR on it.

    `page_number` is 1-indexed to match PDFExtractionResult.pages.
    """

    # Render PDF page as PNG bytes
    png_bytes = render_page_as_image(
        pdf_path,
        page_number,
        zoom=zoom
    )

    # Convert PNG bytes into a PIL Image
    image = Image.open(io.BytesIO(png_bytes))

    # Run Tesseract OCR
    text = pytesseract.image_to_string(image)

    return text.strip()


# ============================================================
# APPLY OCR FALLBACK
# ============================================================

def apply_ocr_fallback(
    pdf_path: str,
    extraction: PDFExtractionResult
) -> PDFExtractionResult:
    """
    Mutates and returns `extraction` in place.

    For every page flagged as `is_likely_scanned`:

    1. Render the page as an image.
    2. Run Tesseract OCR.
    3. Replace the page's empty/near-empty text with OCR output.
    4. Rebuild `full_text`.

    If Tesseract isn't available, the extraction is returned
    unchanged and the application continues without OCR.
    """

    # --------------------------------------------------------
    # Check whether OCR is available
    # --------------------------------------------------------

    if not is_ocr_available():

        logger.warning(
            "Skipping OCR fallback — "
            "Tesseract is not installed on this system."
        )

        return extraction


    # --------------------------------------------------------
    # Find pages that need OCR
    # --------------------------------------------------------

    scanned_pages = [
        p
        for p in extraction.pages
        if p.is_likely_scanned
    ]


    # No scanned pages → nothing to do
    if not scanned_pages:
        return extraction


    logger.info(
        f"Running OCR fallback on "
        f"{len(scanned_pages)} likely-scanned page(s)..."
    )


    # --------------------------------------------------------
    # OCR each scanned page
    # --------------------------------------------------------

    for page in scanned_pages:

        try:

            ocr_text = ocr_page(
                pdf_path,
                page.page_number
            )


            # ------------------------------------------------
            # If OCR successfully recovered text
            # ------------------------------------------------

            if ocr_text:

                logger.info(
                    f"  Page {page.page_number}: "
                    f"OCR recovered "
                    f"{len(ocr_text)} characters"
                )

                page.text = ocr_text

                page.char_count = len(ocr_text)

                # Page is no longer considered unreadable
                page.is_likely_scanned = False


            # ------------------------------------------------
            # OCR returned nothing
            # ------------------------------------------------

            else:

                logger.warning(
                    f"  Page {page.page_number}: "
                    "OCR found no text "
                    "(image may be blank/unreadable)."
                )


        except Exception as e:

            logger.error(
                f"  Page {page.page_number}: "
                f"OCR failed — {e}"
            )


    # --------------------------------------------------------
    # Rebuild full_text after OCR
    # --------------------------------------------------------

    extraction.full_text = "\n\n".join(
        p.text
        for p in extraction.pages
    )


    return extraction