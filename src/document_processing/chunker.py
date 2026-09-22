"""
Document chunking stage of the RAG pipeline.

This module takes a `PDFExtractionResult` (produced by `pdf_extractor.py`,
after OCR fallback has already been applied by `ocr_processor.py`) and
splits each page's text into overlapping, citation-friendly chunks.

Design principle
-----------------
We chunk PAGE BY PAGE, never on the concatenated `full_text`. This means
every chunk we produce already "knows" which page it came from, which is
essential later when the RAG system needs to say:

    "According to the document... (Source: report.pdf, page 5)"

If we chunked `full_text` instead, a single chunk could straddle two pages
and we would lose the ability to cite a single, correct page number.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from src.document_processing.pdf_extractor import PDFExtractionResult
from src.utils.logger import logger


from dataclasses import dataclass, field

# NOTE: adjust this import path to match wherever pdf_extractor.py actually
# lives in your project (e.g. `src.pdf_extractor` or
# `src.pdf_processing.pdf_extractor`). Shown here assuming it sits next to
# this file's parent package.
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Measured in CHARACTERS, not tokens or words. Character counts are the
# simplest thing to reason about at this stage, and they map reasonably
# well to token counts for English text (roughly 1 token ~= 4 characters),
# so 500 characters is a reasonable proxy for "a small-ish paragraph" or
# "~120 tokens" -- small enough that a single chunk stays focused on one
# idea, large enough that it still carries useful context for an LLM.
DEFAULT_CHUNK_SIZE = 500

# 100 characters of overlap (20% of chunk_size) means that a sentence or
# idea that happens to fall right at a chunk boundary is very unlikely to
# be split with NO shared context between the two chunks. Too little
# overlap risks losing context at boundaries; too much overlap wastes
# storage/embedding cost and creates near-duplicate chunks. 20% is a
# common, sensible starting point.
DEFAULT_CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DocumentChunk:
    """
    A single chunk of text extracted from one page of a document.

    Attributes:
        chunk_id: A unique, human-readable identifier for this chunk
            (e.g. "report.pdf_p3_c1"). Useful as a primary key once these
            chunks are stored in a vector database.
        text: The actual chunk text that will be embedded and later
            retrieved.
        page_number: The 1-indexed page number this chunk came from,
            matching `PageResult.page_number` from `pdf_extractor.py`.
        metadata: Extra context carried alongside the chunk for future use
            (citations, filtering, debugging). See `chunk_document()` for
            the exact keys that are populated.
    """

    chunk_id: str
    text: str
    page_number: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Raise ValueError if chunk_size/chunk_overlap are not usable."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size}), otherwise chunks would never "
            "make forward progress."
        )


# ---------------------------------------------------------------------------
# Core word-boundary-aware splitting
# ---------------------------------------------------------------------------


def _split_text_into_windows(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split a single string of text into overlapping windows of
    approximately `chunk_size` characters, breaking on whitespace
    (word boundaries) instead of mid-word whenever possible.

    Approach:
        1. Split the text into words on whitespace.
        2. Greedily accumulate words into the current chunk until adding
           the next word would exceed `chunk_size` characters.
        3. Start the next chunk by "stepping back" into the previous
           chunk's words until we've included roughly `chunk_overlap`
           characters worth of trailing words, then keep accumulating
           forward from there.

    This is intentionally simple (no regex sentence splitting, no NLP
    tokenizer) so it's easy to read, debug, and extend later.
    """
    words = text.split()
    if not words:
        return []

    windows: list[str] = []
    start_idx = 0
    n_words = len(words)

    while start_idx < n_words:
        current_words: list[str] = []
        current_len = 0
        idx = start_idx

        # Greedily add words until we'd exceed chunk_size.
        while idx < n_words:
            word = words[idx]
            # +1 accounts for the joining space (except for the very
            # first word in the chunk).
            added_len = len(word) if not current_words else len(word) + 1
            if current_words and current_len + added_len > chunk_size:
                break
            current_words.append(word)
            current_len += added_len
            idx += 1

        # Edge case: a single word longer than chunk_size. Rather than
        # split the word itself (requirement #5: avoid splitting words
        # mid-way if avoidable), we allow this one chunk to be slightly
        # longer than chunk_size and move on.
        if not current_words:
            current_words = [words[idx]]
            idx += 1

        windows.append(" ".join(current_words))

        if idx >= n_words:
            # We've consumed the whole text; we're done.
            break

        # Figure out where the NEXT chunk should start so that it
        # overlaps with the end of the current chunk by ~chunk_overlap
        # characters, measured by walking backwards from `idx`.
        overlap_len = 0
        back_idx = idx
        while back_idx > start_idx:
            candidate_word = words[back_idx - 1]
            candidate_len = (
                len(candidate_word)
                if overlap_len == 0
                else len(candidate_word) + 1
            )
            if overlap_len + candidate_len > chunk_overlap:
                break
            overlap_len += candidate_len
            back_idx -= 1

        # Guarantee forward progress: the next chunk must start after
        # start_idx, even if chunk_overlap is large relative to word
        # lengths.
        next_start = max(back_idx, start_idx + 1)
        start_idx = next_start

    return windows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    extraction: PDFExtractionResult,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """
    Split a `PDFExtractionResult` into overlapping `DocumentChunk` objects,
    processing each page independently so that page numbers stay accurate
    for citations.

    Args:
        extraction: The result of `extract_text()` (optionally passed
            through `apply_ocr_fallback()`). This function reads
            `extraction.pages`, NOT `extraction.full_text`.
        chunk_size: Target maximum chunk length, in characters.
        chunk_overlap: Number of trailing characters from one chunk that
            should reappear at the start of the next chunk, to preserve
            context across chunk boundaries.

    Returns:
        A list of `DocumentChunk`, in page order, then chunk order within
        each page. Empty or whitespace-only pages produce zero chunks.

    Raises:
        ValueError: If `chunk_size`/`chunk_overlap` are invalid.
    """
    _validate_chunk_params(chunk_size, chunk_overlap)

    file_path = extraction.file_path
    file_name = file_path.split("/")[-1].split("\\")[-1]

    chunks: list[DocumentChunk] = []
    skipped_empty_pages = 0

    for page in extraction.pages:
        page_text = page.text.strip() if page.text else ""

        if not page_text:
            skipped_empty_pages += 1
            logger.info(f"Page {page.page_number} is empty, skipping chunking.")
            continue

        page_windows = _split_text_into_windows(
            page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for chunk_index, window_text in enumerate(page_windows):
            chunk_id = f"{file_name}_p{page.page_number}_c{chunk_index}"
            metadata = {
                "source": file_path,
                "file_name": file_name,
                "page_number": page.page_number,
                "chunk_id": chunk_id,
                "chunk_index_in_page": chunk_index,
                "total_chunks_in_page": len(page_windows),
                "was_ocr_page": page.is_likely_scanned,
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=window_text,
                    page_number=page.page_number,
                    metadata=metadata,
                )
            )

    logger.info(
        f"Chunked '{file_name}': {len(extraction.pages)} pages -> "
        f"{len(chunks)} chunks "
        f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
        f"{skipped_empty_pages} empty pages skipped)"
    )

    return chunks