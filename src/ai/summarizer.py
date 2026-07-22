"""
summarizer.py
-------------
Generates AI-powered summaries and key points from cleaned document text.

DESIGN NOTE ON LONG DOCUMENTS:
LLMs have finite context windows. For Phase 1 we keep this simple:
truncate very long text to a safe character budget before sending it
to the model. From Phase 3 onward, the RAG pipeline handles long
documents properly via chunking + retrieval instead of truncation —
this simple version is fine for most single documents up to
~30-40 pages.
"""

from src.ai.llm_client import LLMClient
from src.utils.logger import logger

# Rough safety budget: ~4 chars/token, leave room for prompt + response.
MAX_INPUT_CHARS = 60_000


def _truncate(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    logger.warning(f"Document text truncated from {len(text)} to {max_chars} chars for LLM call.")
    return text[:max_chars] + "\n\n[...document truncated for length...]"


def generate_summary(text: str, style: str = "concise", client: LLMClient | None = None) -> str:
    """
    style: "concise" (1 paragraph), "detailed" (multi-paragraph), or "executive" (bullet exec summary)
    """
    client = client or LLMClient()
    text = _truncate(text)

    style_instructions = {
        "concise": "Write a single, dense paragraph (4-6 sentences) summarizing the document.",
        "detailed": "Write a thorough multi-paragraph summary (3-5 paragraphs) covering all major sections.",
        "executive": "Write a short executive summary in 4-6 bullet points aimed at a busy decision-maker.",
    }
    instruction = style_instructions.get(style, style_instructions["concise"])

    prompt = f"""{instruction}

Base your summary ONLY on the document text below. Do not invent facts
that aren't in the text.

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"
"""
    logger.info(f"Generating '{style}' summary ({len(text)} chars input)")
    return client.complete(prompt, system="You are an expert document summarizer.", max_tokens=800)


def generate_key_points(text: str, num_points: int = 7, client: LLMClient | None = None) -> list[str]:
    """
    Returns a list of key-point strings extracted from the document.
    """
    client = client or LLMClient()
    text = _truncate(text)

    prompt = f"""Extract the {num_points} most important key points from the document below.

Rules:
- Return ONLY a numbered list, one key point per line.
- Each point should be a single, complete, self-contained sentence.
- Do not include any preamble, headers, or closing remarks — just the list.

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"
"""
    logger.info(f"Generating {num_points} key points")
    raw = client.complete(prompt, system="You extract key points from documents precisely.", max_tokens=800)
    return _parse_numbered_list(raw)


def _parse_numbered_list(raw: str) -> list[str]:
    """Turns '1. Foo\n2. Bar' style output into a clean Python list."""
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    points = []
    for line in lines:
        # strip leading "1.", "1)", "-", "*" etc.
        import re
        cleaned = re.sub(r"^(\d+[\.\)]|[-*])\s*", "", line).strip()
        if cleaned:
            points.append(cleaned)
    return points
