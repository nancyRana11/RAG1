"""
file_utils.py
-------------
Small helpers for saving uploaded files safely and generating
consistent, collision-free filenames / document IDs.
"""

import hashlib
import uuid
from pathlib import Path

from config import UPLOAD_DIR


def generate_doc_id(file_bytes: bytes) -> str:
    """
    Generate a stable, content-based document ID.
    Using a hash (not a random UUID) means if the SAME file is
    uploaded twice, it gets the SAME id — useful later for caching
    embeddings/summaries instead of recomputing them.
    """
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def save_uploaded_file(file_bytes: bytes, original_filename: str) -> Path:
    """
    Persists an uploaded file to disk under data/uploads/ using a
    safe, unique name, and returns the path.
    """
    doc_id = generate_doc_id(file_bytes)
    safe_suffix = Path(original_filename).suffix or ".pdf"
    dest_path = UPLOAD_DIR / f"{doc_id}{safe_suffix}"

    # Avoid re-writing if it's already there (same content -> same doc_id)
    if not dest_path.exists():
        dest_path.write_bytes(file_bytes)

    return dest_path


def new_session_id() -> str:
    """Random ID used for chat/session tracking (Phase 2+)."""
    return str(uuid.uuid4())
