"""
pipeline.py
-----------
Phase 1 orchestrator. This is the ONE function the Streamlit UI calls.
It wires together: PDF extraction -> cleaning -> LLM summary/key
points/metadata, and returns a single object the UI can render.

Keeping this orchestration separate from app.py means:
- app.py stays a thin "view layer" (just UI widgets)
- this logic is testable without Streamlit running at all
- Phase 2+ can reuse/extend this pipeline for chat, JSON extraction, etc.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.document_processing.pdf_extractor import extract_text, PDFExtractionResult
from src.document_processing.text_cleaner import clean_text, basic_stats
from src.ai.summarizer import generate_summary, generate_key_points
from src.ai.metadata_extractor import extract_metadata
from src.ai.llm_client import LLMClient
from src.utils.logger import logger


@dataclass
class ProcessedDocument:
    doc_id: str
    file_name: str
    extraction: PDFExtractionResult
    cleaned_text: str
    stats: dict
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def process_document(
    pdf_path: str | Path,
    doc_id: str,
    summary_style: str = "concise",
    run_summary: bool = True,
    run_key_points: bool = True,
    run_metadata: bool = True,
) -> ProcessedDocument:
    """
    Runs the full Phase 1 pipeline on a single PDF file.
    Each AI step is toggle-able so the UI can let the user skip
    slow/expensive steps if they only want raw text, for example.
    """
    pdf_path = Path(pdf_path)
    logger.info(f"=== Processing document: {pdf_path.name} (doc_id={doc_id}) ===")

    # Step 1: extract raw text from the PDF
    extraction = extract_text(pdf_path)

    if extraction.scanned_page_ratio > 0.5:
        logger.warning(
            f"{pdf_path.name} looks like a scanned document "
            f"({extraction.scanned_page_ratio:.0%} pages with little/no text). "
            "OCR support is added in Phase 2 — text below may be incomplete."
        )

    # Step 2: clean the text
    cleaned = clean_text(extraction.full_text)
    stats = basic_stats(cleaned)

    doc = ProcessedDocument(
        doc_id=doc_id,
        file_name=pdf_path.name,
        extraction=extraction,
        cleaned_text=cleaned,
        stats=stats,
    )

    if not cleaned.strip():
        logger.warning("No text extracted — skipping AI steps (nothing to send to the LLM).")
        return doc

    # Reuse a single LLMClient across steps (avoids re-initializing the SDK 3x)
    client = LLMClient()

    # Step 3: AI summary
    if run_summary:
        doc.summary = generate_summary(cleaned, style=summary_style, client=client)

    # Step 4: AI key points
    if run_key_points:
        doc.key_points = generate_key_points(cleaned, client=client)

    # Step 5: AI metadata
    if run_metadata:
        doc.metadata = extract_metadata(cleaned, client=client)

    logger.info(f"=== Finished processing: {pdf_path.name} ===")
    return doc
