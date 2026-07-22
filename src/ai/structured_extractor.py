"""
structured_extractor.py
------------------------
Extracts structured data from a document according to a schema the
USER defines at runtime (e.g. "invoice_number", "total_amount",
"due_date" for an invoice; "candidate_name", "years_experience" for a
resume). This is different from `metadata_extractor.py` (Phase 1),
which extracts a FIXED, generic schema for every document.

Two ways to specify what to extract:
1. A list of field names (simple) -> we infer types generically.
2. A dict of {field_name: description} (more precise extraction).
"""

import json
import re

from src.ai.llm_client import LLMClient
from src.utils.logger import logger

MAX_INPUT_CHARS = 30_000


def extract_structured_data(
    text: str,
    fields: list[str] | dict[str, str],
    client: LLMClient | None = None,
) -> dict:
    """
    Parameters
    ----------
    text : cleaned document text
    fields : either
        - ["invoice_number", "total_amount", "due_date"]
        - {"invoice_number": "The unique invoice ID, e.g. INV-2024-001",
           "total_amount": "The final total amount due, as a number"}
    client : optional shared LLMClient

    Returns
    -------
    dict mapping each requested field -> extracted value (or null/None
    if not found in the document).
    """
    client = client or LLMClient()
    truncated = text[:MAX_INPUT_CHARS]

    field_spec = _build_field_spec(fields)

    prompt = f"""Extract the following fields from the document text below.
Return ONLY a valid JSON object with EXACTLY these keys (no extra keys,
no markdown fences, no commentary):

{field_spec}

Rules:
- If a field's value is not present in the document, set it to null.
- Do not guess or invent values that aren't supported by the text.
- Return numbers as numbers (not strings) where the field is clearly numeric.

DOCUMENT TEXT:
\"\"\"
{truncated}
\"\"\"
"""
    logger.info(f"Extracting structured fields: {list(_normalize_fields(fields).keys())}")
    raw = client.complete(
        prompt,
        system="You are a precise structured-data extraction engine. You only output valid JSON.",
        max_tokens=800,
        temperature=0.0,
    )
    result = _safe_parse_json(raw)

    # Ensure every requested field is present in the result, even if the
    # LLM's JSON was missing one — keeps downstream code (e.g. a table UI)
    # from KeyError-ing.
    normalized = _normalize_fields(fields)
    for field_name in normalized:
        result.setdefault(field_name, None)

    return result


def _normalize_fields(fields: list[str] | dict[str, str]) -> dict[str, str]:
    if isinstance(fields, dict):
        return fields
    return {f: "" for f in fields}


def _build_field_spec(fields: list[str] | dict[str, str]) -> str:
    normalized = _normalize_fields(fields)
    lines = []
    for name, description in normalized.items():
        if description:
            lines.append(f'  "{name}": <{description}>')
        else:
            lines.append(f'  "{name}": <value or null>')
    return "{\n" + ",\n".join(lines) + "\n}"


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
        logger.error(f"Failed to parse structured-extraction JSON: {raw[:300]}")
        return {"error": "Failed to parse structured output", "raw_output": raw}
