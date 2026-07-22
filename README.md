# LLM-Powered Document Intelligence System

A local Python + Streamlit application that lets you upload PDFs, get AI
summaries/key points/metadata, chat with your documents, compare multiple
documents, and (eventually) run semantic search over a document library
via a full RAG pipeline — orchestrated by an AI agent.

This README covers **Phase 1** only. Later phases (OCR, chat, RAG, agents)
will extend this same document as they're built.

---

## 1. Full Project Architecture (all 4 phases, for orientation)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI (app.py)                       │
│   Upload | Summary | Chat | Compare | Search | Agent Console        │
└───────────────────────────────┬───────────────────────────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 │                                 │
        ┌────────▼─────────┐            ┌──────────▼──────────┐
        │ Document Pipeline │            │   Orchestrator Agent │  (Phase 4)
        │  (Phase 1 core)   │            │  routes to tools:    │
        └────────┬──────────┘            │  summarize / search /│
                 │                       │  qa / extract / cmp   │
   ┌─────────────┼──────────────┐        └──────────┬───────────┘
   │             │              │                    │
┌──▼───┐    ┌────▼────┐   ┌─────▼─────┐        ┌─────▼──────┐
│ PDF  │    │  Text   │   │    AI     │        │  RAG Layer  │ (Phase 3)
│Extract│──▶│ Cleaner │──▶│(LLM calls)│        │ chunk→embed │
│(PyMuPDF)   │         │   │summary/   │        │→ vector DB  │
│ +OCR  │   │         │   │metadata/  │        │→ retrieve   │
│(Ph.2) │   │         │   │key points │        │→ generate   │
└──────┘    └─────────┘   └───────────┘        └────────────┘
```

- **Phase 1 (this delivery):** upload → extract → clean → summarize → key points → metadata
- **Phase 2:** OCR fallback, chat with a single PDF, structured JSON extraction, multi-doc upload & compare, general Q&A
- **Phase 3:** chunking, embeddings, ChromaDB/FAISS vector store, semantic search, conversation memory, full RAG
- **Phase 4:** LangGraph orchestrator agent that decides which tool(s) to call and chains multi-step reasoning

---

## 2. Folder Structure (Phase 1 state)

```
doc_intelligence_system/
├── app.py                              # Streamlit UI (entry point)
├── config.py                           # Central config (env vars, paths)
├── requirements.txt
├── .env.example                        # Copy to .env and fill in your API key
├── README.md
│
├── src/
│   ├── document_processing/
│   │   ├── pdf_extractor.py            # PyMuPDF text extraction
│   │   ├── text_cleaner.py             # Cleaning/preprocessing
│   │   └── pipeline.py                 # Orchestrates extract→clean→AI steps
│   │
│   ├── ai/
│   │   ├── llm_client.py               # Provider-agnostic Anthropic/OpenAI wrapper
│   │   ├── summarizer.py               # Summary + key points generation
│   │   └── metadata_extractor.py       # Structured JSON metadata via LLM
│   │
│   └── utils/
│       ├── logger.py                   # Central logging (console + file)
│       └── file_utils.py               # Safe upload saving, ID generation
│
├── data/
│   ├── uploads/                        # Saved uploaded PDFs (gitignored)
│   ├── processed/                      # Logs, cached outputs
│   └── vector_store/                   # (used from Phase 3 onward)
│
└── tests/
    ├── test_pdf_extractor.py
    └── test_text_cleaner.py
```

**Folders NOT yet present** (added in later phases, so you know what's coming):
`src/document_processing/ocr_processor.py` (Phase 2), `src/rag/` (Phase 3),
`src/agents/` (Phase 4).

---

## 3. Installation Steps

### Prerequisites
- Python 3.10+ (3.12 recommended)
- pip
- (Phase 2 only, not needed yet) Tesseract OCR installed at the OS level

### Steps

```bash
# 1. Clone/unzip the project, then cd into it
cd doc_intelligence_system

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up your environment file
cp .env.example .env
# then open .env and paste in your ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 6. Run the app
streamlit run app.py
```

Your browser should open automatically to `http://localhost:8501`.

---

## 4. Environment Setup Details

`.env` variables explained:

| Variable | Purpose | Phase |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` or `openai` | 1 |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Claude credentials + model name | 1 |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI credentials + model name (alt. provider) | 1 |
| `MAX_UPLOAD_MB` | Upload size guard | 1 |
| `LOG_LEVEL` | Console log verbosity | 1 |
| `EMBEDDING_PROVIDER`, `CHROMA_PERSIST_DIR`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | Reserved for RAG | 3 |

> **Note on model names:** Anthropic and OpenAI periodically release new
> model versions/strings. `ANTHROPIC_MODEL` defaults to `claude-sonnet-5`
> in `.env.example` — check https://docs.claude.com for the current
> recommended model string if you hit a "model not found" error.

---

## 5. Required Libraries (Phase 1)

| Library | Why |
|---|---|
| `streamlit` | Web UI |
| `PyMuPDF` (`fitz`) | Fast, high-fidelity PDF text extraction |
| `anthropic` | Claude API SDK |
| `openai` | OpenAI API SDK (alternative provider) |
| `python-dotenv` | Loads `.env` into environment variables |
| `pydantic` | Data validation (used more heavily from Phase 2) |
| `tenacity` | Retry/backoff for flaky API calls |
| `loguru` | Clean, readable logging to console + file |
| `pytest` | Testing |

Later phases add: `pytesseract`, `pdf2image` (Phase 2 OCR),
`chromadb` or `faiss-cpu`, `sentence-transformers`, `langchain`,
`langchain-community`, `langgraph` (Phases 3-4).

---

## 6. Database Design

Phase 1 has **no database** — processed results live in Streamlit's
`st.session_state` (in-memory, per browser session) and raw files are
saved to `data/uploads/`.

From **Phase 3 onward**, we introduce:
- **ChromaDB** (embedded, file-based vector database) storing:
  - `documents`: chunk text
  - `embeddings`: vector representations
  - `metadatas`: `{doc_id, chunk_index, page_number, source_filename}`
  - `ids`: `{doc_id}_{chunk_index}`
- This lets semantic search and RAG retrieval filter by document and
  reconstruct exactly where a retrieved chunk came from.

We are **not** introducing a relational database (Postgres/SQLite) in
this build since ChromaDB's metadata store is sufficient for the scope
here — this is called out under "Future Enhancements" as an option for
production hardening (e.g., a proper `documents` table for multi-user
support).

---

## 7. Phase 1 Data Flow

```
User uploads PDF (Streamlit file_uploader)
        │
        ▼
save_uploaded_file()  ──▶ data/uploads/{sha256_hash}.pdf
        │
        ▼
extract_text()  (PyMuPDF)
        │  → per-page text, PDF metadata, "looks scanned?" flags
        ▼
clean_text()
        │  → fixes hyphenation, strips page numbers, collapses whitespace
        ▼
   ┌────┴─────┬──────────────┬───────────────┐
   ▼          ▼              ▼               │
generate_   generate_    extract_             │
summary()   key_points() metadata()           │
   │          │              │                │
   └──────────┴──────────────┴────────────────┘
                     │
                     ▼
           ProcessedDocument object
                     │
                     ▼
        Rendered in Streamlit tabs (Summary / Key Points /
        Metadata / Stats / Raw Text) + stored in session_state
```

---

## 8. Testing Strategy

- **Unit tests** (`tests/`) cover pure-logic modules that don't need an
  API key: `pdf_extractor.py` (using PDFs generated on-the-fly with
  PyMuPDF itself — no fixture files needed) and `text_cleaner.py`.
- **LLM-calling modules** (`summarizer.py`, `metadata_extractor.py`,
  `llm_client.py`) are intentionally structured so `LLMClient` can be
  swapped for a mock/fake in tests (dependency passed as a parameter,
  not hardcoded) — this pattern is used more heavily once we add CI in
  a later phase.
- Run everything with:
  ```bash
  pytest tests/ -v
  ```

**Phase 1 test results:** 12/12 passing (verified during this build).

---

## 9. Roadmap / What's Next

| Phase | Adds | Status |
|---|---|---|
| **1** | Upload, extract, clean, summarize, key points, metadata | ✅ Delivered |
| **2** | OCR, chat with PDF, structured JSON extraction, multi-doc upload & comparison, Q&A | ⏳ Next |
| **3** | Chunking, embeddings, ChromaDB/FAISS, semantic search, memory, full RAG | ⏳ Planned |
| **4** | LangGraph orchestrator agent, tool routing, multi-step reasoning | ⏳ Planned |

---

## 10. Deployment Strategy (preview — expanded in a later phase)

For local use: `streamlit run app.py` is sufficient. For sharing with a
small team: Streamlit Community Cloud, a Docker container behind a
reverse proxy, or an internal VM with `streamlit run app.py --server.port 8501
--server.address 0.0.0.0`. A `Dockerfile` and docker-compose setup will be
added once the RAG layer (Phase 3) introduces a persistent vector store
that needs a mounted volume.

---

## 11. Future Enhancements (preview)

- Multi-user support with a proper database (Postgres) instead of
  session-state
- Streaming LLM responses in the UI instead of waiting for full completion
- Support for .docx, .txt, and image-only uploads
- Cost/usage tracking per document processed
- Authentication for shared/team deployments
