"""
Unit tests for text_cleaner.py.
Run with: pytest tests/test_text_cleaner.py -v
"""

import sys
from pathlib import Path

# Allow running `pytest` from the project root without package installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_processing.text_cleaner import clean_text, basic_stats


def test_fixes_hyphenated_line_breaks():
    raw = "This is an imple-\nmentation detail."
    result = clean_text(raw)
    assert "implementation" in result
    assert "imple-" not in result


def test_removes_standalone_page_numbers():
    raw = "Some content here.\n\n3\n\nMore content."
    result = clean_text(raw)
    assert "\n3\n" not in result


def test_removes_page_x_of_y_lines():
    raw = "Content.\nPage 2 of 10\nMore content."
    result = clean_text(raw)
    assert "Page 2 of 10" not in result


def test_collapses_excess_blank_lines():
    raw = "Line one.\n\n\n\n\nLine two."
    result = clean_text(raw)
    assert "\n\n\n" not in result


def test_basic_stats_counts_words_and_chars():
    text = "one two three four five"
    stats = basic_stats(text)
    assert stats["word_count"] == 5
    assert stats["char_count"] == len(text)


def test_clean_text_strips_leading_trailing_whitespace():
    raw = "   \n  Hello world  \n   "
    result = clean_text(raw)
    assert result == "Hello world"
