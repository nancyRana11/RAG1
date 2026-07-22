"""
config.py
---------
Single source of truth for all configuration values.
Every other module reads settings from here instead of calling
os.getenv() directly — that way, if you ever need to change a
default, you change it in ONE place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a .env file (if present) into the process environment.
load_dotenv()

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"   # used from Phase 3 onward

for _dir in (UPLOAD_DIR, PROCESSED_DIR, VECTOR_STORE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# LLM provider settings
# ---------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------
# App settings
# ---------------------------------------------------------------
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------
# RAG settings (used starting Phase 3, defined now so config is stable)
# ---------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
CHROMA_PERSIST_DIR = str(VECTOR_STORE_DIR)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def validate_config() -> list[str]:
    """
    Returns a list of human-readable problems with the current config.
    Call this at app startup so the user gets a clear error instead of
    a confusing stack trace three modules deep.
    """
    problems = []
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        problems.append("ANTHROPIC_API_KEY is not set (required because LLM_PROVIDER=anthropic).")
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        problems.append("OPENAI_API_KEY is not set (required because LLM_PROVIDER=openai).")
    if LLM_PROVIDER not in ("anthropic", "openai","gemini"):
        problems.append(f"LLM_PROVIDER must be 'anthropic' or 'openai', got '{LLM_PROVIDER}'.")
    return problems
