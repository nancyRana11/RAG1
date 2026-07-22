"""
text_cleaner.py
----------------
Cleans raw text extracted from PDFs before it goes to the LLM or into
the RAG pipeline. PDF text extraction is messy: broken hyphenation,
weird whitespace, repeated headers/footers, page-number noise, etc.
Cleaning this up measurably improves summary quality and chunk quality.
"""

import re


def clean_text(raw_text: str) -> str:
    """
    Applies a pipeline of cleaning steps. Order matters — each step
    assumes the previous one already ran.
    """
    text = raw_text

    text = _fix_hyphenated_line_breaks(text)
    text = _normalize_whitespace(text)
    text = _remove_common_page_artifacts(text)
    text = _collapse_blank_lines(text)

    return text.strip()


def _fix_hyphenated_line_breaks(text: str) -> str:
    """
    PDFs often break words across lines with a hyphen, e.g.:
        "The quick brown fox jumped over the lazy dog in the
         imple-
         mentation of the algorithm"
    This merges "imple-\nmentation" -> "implementation".
    """
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    # Collapse runs of spaces/tabs (but keep newlines intact for now)
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _remove_common_page_artifacts(text: str) -> str:
    """
    Strips lines that are almost certainly headers/footers/page numbers,
    e.g. a line that is JUST a number, or "Page 3 of 12".
    """
    cleaned_lines = []
    page_number_pattern = re.compile(r"^\s*(page\s+)?\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE)

    for line in text.split("\n"):
        if page_number_pattern.match(line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _collapse_blank_lines(text: str) -> str:
    """Turns 3+ consecutive newlines into exactly 2 (one blank line)."""
    return re.sub(r"\n{3,}", "\n\n", text)


def basic_stats(text: str) -> dict:
    """Quick stats used in the UI and for sanity-checking extraction quality."""
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
        "estimated_tokens": int(len(words) * 1.3),  # rough heuristic, ~1.3 tokens/word
        "line_count": text.count("\n") + 1,
    }
