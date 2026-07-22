"""
metadata_extractor.py
----------------------
Generates structured "document metadata" using the LLM: title, document
type, topics, entities, sentiment/tone, and a language guess. This is
distinct from the raw PDF metadata (author/creation date from the PDF
file itself) which pdf_extractor.py already pulls out for free.

Output is strict JSON so the Streamlit UI can render it as a table and
so later phases (structured extraction, comparison) can rely on a
predictable schema.
"""

import json
import re

from src.ai.llm_client import LLMClient
from src.utils.logger import logger

MAX_INPUT_CHARS = 20_000  # metadata needs less context than a full summary

METADATA_SCHEMA_HINT = """{
  "title": "string - best-guess title of the document",
  "document_type": "string - e.g. 'invoice', 'research paper', 'contract', 'report', 'resume', 'other'",
  "topics": ["list", "of", "3-6 short topic tags"],
  "key_entities": {
    "people": ["names mentioned"],
    "organizations": ["organizations mentioned"],
    "dates": ["important dates mentioned"],
    "locations": ["locations mentioned"]
  },
  "tone": "string - e.g. 'formal', 'technical', 'persuasive', 'neutral'",
  "language": "string - detected primary language, e.g. 'English'",
  "estimated_reading_time_minutes": "integer"
}"""


def extract_metadata(text: str, client: LLMClient | None = None) -> dict:
    client = client or LLMClient()
    truncated = text[:MAX_INPUT_CHARS]

    prompt = f"""Analyze the document text below and return ONLY a valid JSON object
matching exactly this schema (no markdown fences, no commentary, no extra keys):

{METADATA_SCHEMA_HINT}

If a field cannot be determined, use an empty string, empty list, or
empty object as appropriate — never invent information not supported
by the text.

DOCUMENT TEXT:
\"\"\"
{truncated}
\"\"\"
"""
    logger.info("Extracting document metadata via LLM")
    raw = client.complete(
        prompt,
        system="You are a precise information-extraction engine. You only output valid JSON.",
        max_tokens=700,
        temperature=0.0,
    )
    return _safe_parse_json(raw)


def _safe_parse_json(raw: str) -> dict:
    """
    LLMs occasionally wrap JSON in ```json fences or add stray text.
    This defensively extracts the JSON object before parsing.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: grab the first {...} block found
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.error(f"Failed to parse metadata JSON from LLM output: {raw[:300]}")
        return {"error": "Failed to parse structured metadata", "raw_output": raw}
