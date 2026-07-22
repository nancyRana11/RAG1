"""
comparison.py
--------------
Compares 2+ documents and produces a structured comparison: shared
themes, key differences, and a side-by-side summary table.

Like qa.py, this uses context stuffing (each document's cleaned text,
truncated) rather than retrieval — fine for a handful of documents of
reasonable length, revisited if needed once RAG (Phase 3) is in place.
"""

import json
import re

from src.ai.llm_client import LLMClient
from src.utils.logger import logger

MAX_CHARS_PER_DOC = 15_000  # budget per document so N docs still fit in context

COMPARISON_SCHEMA_HINT = """{
  "overview": "string - 2-3 sentence summary of how these documents relate to each other",
  "shared_themes": ["list of themes/topics common to multiple documents"],
  "key_differences": ["list of notable differences between the documents"],
  "per_document_summary": {
    "<file_name>": "one sentence summary of this specific document"
  },
  "recommendation": "string - optional, a suggested takeaway or next step; empty string if not applicable"
}"""


def compare_documents(documents: dict[str, str], client: LLMClient | None = None) -> dict:
    """
    Parameters
    ----------
    documents : dict mapping file_name -> cleaned document text
                (needs at least 2 entries)
    """
    if len(documents) < 2:
        raise ValueError("compare_documents needs at least 2 documents.")

    client = client or LLMClient()

    doc_blocks = []
    for name, text in documents.items():
        truncated = text[:MAX_CHARS_PER_DOC]
        doc_blocks.append(f'--- DOCUMENT: "{name}" ---\n{truncated}')
    combined = "\n\n".join(doc_blocks)

    prompt = f"""Compare the following {len(documents)} documents. Return ONLY a valid
JSON object matching exactly this schema (no markdown fences, no commentary):

{COMPARISON_SCHEMA_HINT}

The "per_document_summary" object must have exactly one key per document,
using the EXACT document names given below.

{combined}
"""
    logger.info(f"Comparing {len(documents)} documents: {list(documents.keys())}")
    raw = client.complete(
        prompt,
        system="You are an expert analyst who compares documents precisely and objectively.",
        max_tokens=1500,
        temperature=0.2,
    )
    return _safe_parse_json(raw)


def _safe_parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.error(f"Failed to parse comparison JSON: {raw[:300]}")
        return {"error": "Failed to parse comparison output", "raw_output": raw}
