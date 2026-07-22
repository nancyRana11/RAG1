"""
Unit tests for pdf_extractor.py.
These tests generate a tiny throwaway PDF with PyMuPDF itself so no
external fixture file is needed.
Run with: pytest tests/test_pdf_extractor.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz
import pytest

from src.document_processing.pdf_extractor import extract_text


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """Creates a simple 2-page PDF with known text on each page."""
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Hello World. This is page one.")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "This is page two with different content.")

    doc.set_metadata({"title": "Test Document", "author": "Pytest"})
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extracts_correct_number_of_pages(sample_pdf):
    result = extract_text(sample_pdf)
    assert result.num_pages == 2
    assert len(result.pages) == 2


def test_extracts_expected_text_content(sample_pdf):
    result = extract_text(sample_pdf)
    assert "Hello World" in result.pages[0].text
    assert "page two" in result.pages[1].text


def test_full_text_combines_all_pages(sample_pdf):
    result = extract_text(sample_pdf)
    assert "Hello World" in result.full_text
    assert "page two" in result.full_text


def test_pdf_metadata_is_captured(sample_pdf):
    result = extract_text(sample_pdf)
    assert result.pdf_metadata.get("title") == "Test Document"


def test_pages_with_real_text_are_not_flagged_scanned(sample_pdf):
    result = extract_text(sample_pdf)
    assert result.scanned_page_ratio == 0.0


def test_blank_pdf_is_flagged_as_scanned(tmp_path):
    """A PDF page with NO text at all should be flagged as likely-scanned."""
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()  # blank page, no text inserted
    doc.save(pdf_path)
    doc.close()

    result = extract_text(pdf_path)
    assert result.pages[0].is_likely_scanned is True
    assert result.scanned_page_ratio == 1.0
