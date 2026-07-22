"""
pdf_extractor.py
----------------
Responsible for ONE thing: turning a PDF file into raw text + basic
per-page metadata using PyMuPDF (imported as `fitz`).

Why PyMuPDF over other libraries (pdfplumber, PyPDF2)?
- Much faster on large PDFs
- Better text-extraction fidelity (keeps reading order more reliably)
- Gives us page images for free later (needed for OCR in Phase 2)

NOTE: This module does NOT do OCR. If a PDF page has no extractable
text (i.e. it's a scanned image), `extract_text` will return an empty
string for that page. Phase 2 adds `ocr_processor.py` to handle that
case by rendering the page to an image and running Tesseract on it.
"""

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from src.utils.logger import logger


@dataclass
class PageResult:
    page_number: int          # 1-indexed, human friendly
    text: str
    char_count: int
    is_likely_scanned: bool   # True if almost no text was found on this page


@dataclass
class PDFExtractionResult:
    file_path: str
    num_pages: int
    full_text: str
    pages: list[PageResult] = field(default_factory=list)
    pdf_metadata: dict = field(default_factory=dict)  # title/author/etc from the PDF itself

    @property
    def scanned_page_ratio(self) -> float:
        if not self.pages:
            return 0.0
        scanned = sum(1 for p in self.pages if p.is_likely_scanned)
        return scanned / len(self.pages)


# A page with fewer than this many characters is treated as "likely scanned"
# (i.e. probably an image with no embedded text layer).
MIN_CHARS_FOR_TEXT_PAGE = 20


def extract_text(pdf_path: str | Path) -> PDFExtractionResult:
    """
    Extract text from every page of a PDF.

    Parameters
    ----------
    pdf_path : path to a .pdf file on disk

    Returns
    -------
    PDFExtractionResult with full_text, per-page breakdown, and metadata.
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Extracting text from PDF: {pdf_path.name}")

    doc = fitz.open(pdf_path)
    pages: list[PageResult] = []
    all_text_parts: list[str] = []

    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.strip()

            pages.append(
                PageResult(
                    page_number=i,
                    text=text,
                    char_count=len(text),
                    is_likely_scanned=len(text) < MIN_CHARS_FOR_TEXT_PAGE,
                )
            )
            all_text_parts.append(text)

        pdf_metadata = doc.metadata or {}
        result = PDFExtractionResult(
            file_path=str(pdf_path),
            num_pages=len(doc),
            full_text="\n\n".join(all_text_parts),
            pages=pages,
            pdf_metadata=pdf_metadata,
        )
    finally:
        doc.close()

    logger.info(
        f"Extracted {len(result.full_text)} characters across {result.num_pages} pages "
        f"({result.scanned_page_ratio:.0%} pages look scanned)"
    )
    return result


def render_page_as_image(pdf_path: str | Path, page_number: int, zoom: float = 2.0):
    """
    Renders a single page to a PIL-compatible image (PNG bytes).
    Not used in Phase 1 — this is here so Phase 2's OCR module can
    reuse the same PDF-opening logic without duplicating code.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    finally:
        doc.close()
