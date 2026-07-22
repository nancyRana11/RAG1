"""
app.py
------
Streamlit entry point. Run with:

    streamlit run app.py

PHASE 2 SCOPE adds, on top of Phase 1:
- OCR fallback for scanned PDFs (automatic, happens during processing)
- Multi-document upload
- Chat with a document (conversational, with memory)
- Structured JSON extraction with a user-defined schema
- Multi-document comparison

The app is organized as a single script with a sidebar page switcher
(simplest form of multi-page navigation for Streamlit). Phase 3/4 will
add a "Search" and "Agent" page using the same pattern.
"""

import json

import streamlit as st

from config import validate_config, MAX_UPLOAD_MB
from src.utils.file_utils import save_uploaded_file, generate_doc_id
from src.document_processing.pipeline import process_document
from src.document_processing.ocr_processor import is_ocr_available
from src.ai.qa import ChatSession
from src.ai.structured_extractor import extract_structured_data
from src.ai.comparison import compare_documents
from src.utils.logger import logger

st.set_page_config(page_title="Document Intelligence System", page_icon="📄", layout="wide")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = {}   # doc_id -> ProcessedDocument
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}    # doc_id -> ChatSession

# ------------------------------------------------------------------
# Sidebar: config status + page navigation
# ------------------------------------------------------------------
st.sidebar.title("📄 Doc Intelligence")
st.sidebar.caption("Phase 2: Document Intelligence")

config_problems = validate_config()
if config_problems:
    st.sidebar.error("Configuration problems:\n\n" + "\n".join(f"- {p}" for p in config_problems))
    st.sidebar.info("Copy `.env.example` to `.env` and fill in your API key, then restart the app.")

if not is_ocr_available():
    st.sidebar.warning("Tesseract OCR not found — scanned PDFs won't be recoverable. See README.")

page = st.sidebar.radio(
    "Navigate",
    ["📤 Upload & Analyze", "💬 Chat with Document", "🏷️ Extract Structured Data", "📊 Compare Documents"],
)

st.sidebar.divider()
st.sidebar.caption(
    "Coming in Phase 3: semantic search across all your documents and "
    "proper retrieval-augmented chat. Phase 4: an AI agent that routes "
    "between all these tools automatically."
)


def _doc_picker(label: str, key: str, multiple: bool = False):
    """Shared helper: lets the user pick from already-processed documents."""
    if not st.session_state.processed_docs:
        st.info("No documents processed yet. Go to **📤 Upload & Analyze** first.")
        return None

    options = {
        doc_id: doc.file_name for doc_id, doc in st.session_state.processed_docs.items()
    }
    if multiple:
        return st.multiselect(label, options=list(options.keys()), format_func=lambda d: options[d], key=key)
    else:
        return st.selectbox(label, options=list(options.keys()), format_func=lambda d: options[d], key=key)


# ==================================================================
# PAGE 1: Upload & Analyze
# ==================================================================
if page == "📤 Upload & Analyze":
    st.title("Upload & Analyze Documents")
    st.write("Upload one or more PDFs. Each is extracted, OCR'd if needed, and analyzed independently.")

    with st.expander("⚙️ Processing options", expanded=False):
        summary_style = st.selectbox("Summary style", ["concise", "detailed", "executive"], index=0)
        c1, c2, c3, c4 = st.columns(4)
        run_summary = c1.checkbox("Summary", value=True)
        run_key_points = c2.checkbox("Key points", value=True)
        run_metadata = c3.checkbox("Metadata", value=True)
        run_ocr = c4.checkbox("OCR fallback", value=True)

    uploaded_files = st.file_uploader(
        "Upload PDF document(s)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            size_mb = len(file_bytes) / (1024 * 1024)
            doc_id = generate_doc_id(file_bytes)
            already_processed = doc_id in st.session_state.processed_docs

            with st.container(border=True):
                cols = st.columns([4, 1, 1])
                cols[0].write(f"**{uploaded_file.name}**  ·  {size_mb:.2f} MB  ·  `{doc_id}`")

                if size_mb > MAX_UPLOAD_MB:
                    st.error(f"Exceeds {MAX_UPLOAD_MB} MB limit — skipped.")
                    continue

                status = cols[1]
                status.success("✅ Processed") if already_processed else status.write("⏳ Not processed")

                process_clicked = cols[2].button(
                    "Re-process" if already_processed else "Process",
                    key=f"process_{doc_id}",
                )

                if process_clicked or not already_processed:
                    if config_problems:
                        st.warning("Fix configuration problems in the sidebar before processing.")
                    else:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            try:
                                saved_path = save_uploaded_file(file_bytes, uploaded_file.name)
                                doc = process_document(
                                    saved_path,
                                    doc_id=doc_id,
                                    summary_style=summary_style,
                                    run_summary=run_summary,
                                    run_key_points=run_key_points,
                                    run_metadata=run_metadata,
                                    run_ocr=run_ocr,
                                )
                                st.session_state.processed_docs[doc_id] = doc
                                st.rerun()
                            except Exception as e:
                                logger.exception("Processing failed")
                                st.error(f"Processing failed: {e}")

    st.divider()

    if st.session_state.processed_docs:
        st.subheader("Processed Documents")
        selected_id = _doc_picker("View results for:", key="view_doc")
        doc = st.session_state.processed_docs.get(selected_id) if selected_id else None

        if doc:
            if doc.ocr_applied:
                st.info("🔍 OCR was applied to recover text from scanned page(s) in this document.")
            if doc.extraction.scanned_page_ratio > 0.3 and not doc.ocr_applied:
                st.warning(
                    f"⚠️ {doc.extraction.scanned_page_ratio:.0%} of pages still look scanned/empty "
                    "(OCR may have been disabled or unavailable)."
                )

            tabs = st.tabs(["📝 Summary", "🔑 Key Points", "🏷️ Metadata", "📊 Stats", "📄 Raw Text"])

            with tabs[0]:
                st.write(doc.summary or "_Not generated._")
            with tabs[1]:
                if doc.key_points:
                    for i, point in enumerate(doc.key_points, start=1):
                        st.markdown(f"**{i}.** {point}")
                else:
                    st.write("_Not generated._")
            with tabs[2]:
                st.json(doc.metadata or {})
                st.caption("PDF file metadata:")
                st.json(doc.extraction.pdf_metadata)
            with tabs[3]:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pages", doc.extraction.num_pages)
                c2.metric("Words", doc.stats["word_count"])
                c3.metric("Characters", doc.stats["char_count"])
                c4.metric("Est. tokens", doc.stats["estimated_tokens"])
            with tabs[4]:
                st.text_area("Full text", doc.cleaned_text, height=400)
                st.download_button("⬇ Download text", doc.cleaned_text, file_name=f"{doc.file_name}.txt")
    else:
        st.info("👆 Upload PDFs to get started.")


# ==================================================================
# PAGE 2: Chat with Document
# ==================================================================
elif page == "💬 Chat with Document":
    st.title("Chat with a Document")
    st.caption(
        "Conversational Q&A grounded in the document's text. "
        "(Uses full-document context for now — Phase 3 adds proper retrieval for large documents.)"
    )

    selected_id = _doc_picker("Choose a document to chat with:", key="chat_doc")

    if selected_id:
        doc = st.session_state.processed_docs[selected_id]

        if selected_id not in st.session_state.chat_sessions:
            st.session_state.chat_sessions[selected_id] = ChatSession(
                doc_id=selected_id, document_text=doc.cleaned_text
            )
        session: ChatSession = st.session_state.chat_sessions[selected_id]

        col1, col2 = st.columns([5, 1])
        col1.write(f"Chatting with **{doc.file_name}**")
        if col2.button("🗑️ Clear chat"):
            session.reset()
            st.rerun()

        for msg in session.history:
            with st.chat_message(msg.role):
                st.write(msg.content)

        question = st.chat_input("Ask a question about this document...")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = session.ask(question)
                        st.write(answer)
                    except Exception as e:
                        logger.exception("Chat failed")
                        st.error(f"Something went wrong: {e}")


# ==================================================================
# PAGE 3: Extract Structured Data
# ==================================================================
elif page == "🏷️ Extract Structured Data":
    st.title("Extract Structured Data")
    st.caption("Define the fields you want pulled out of a document, and get back structured JSON.")

    selected_id = _doc_picker("Choose a document:", key="extract_doc")

    PRESETS = {
        "Custom": {},
        "Invoice": {
            "invoice_number": "the unique invoice ID",
            "invoice_date": "the date the invoice was issued",
            "due_date": "the payment due date",
            "total_amount": "the final total amount due, as a number",
            "vendor_name": "the name of the company issuing the invoice",
        },
        "Resume": {
            "candidate_name": "full name of the candidate",
            "email": "contact email address",
            "years_experience": "total years of professional experience, as a number",
            "skills": "list of key technical/professional skills",
            "most_recent_employer": "most recent company worked for",
        },
        "Contract": {
            "parties": "list of the parties entering the contract",
            "effective_date": "the date the contract becomes effective",
            "termination_date": "the date the contract ends, if specified",
            "governing_law": "the jurisdiction/governing law clause",
        },
    }

    preset_name = st.selectbox("Start from a preset (or choose Custom)", list(PRESETS.keys()))

    if preset_name == "Custom":
        raw_fields = st.text_area(
            "Field names (comma-separated)",
            placeholder="e.g. project_name, budget, deadline, stakeholders",
        )
        fields = [f.strip() for f in raw_fields.split(",") if f.strip()]
    else:
        st.write("Fields to extract:")
        st.json(PRESETS[preset_name])
        fields = PRESETS[preset_name]

    if selected_id and fields and st.button("Extract", type="primary"):
        doc = st.session_state.processed_docs[selected_id]
        with st.spinner("Extracting structured data..."):
            try:
                result = extract_structured_data(doc.cleaned_text, fields)
                st.subheader("Result")
                st.json(result)
                st.download_button(
                    "⬇ Download JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"{doc.file_name}.extracted.json",
                    mime="application/json",
                )
            except Exception as e:
                logger.exception("Structured extraction failed")
                st.error(f"Extraction failed: {e}")


# ==================================================================
# PAGE 4: Compare Documents
# ==================================================================
elif page == "📊 Compare Documents":
    st.title("Compare Documents")
    st.caption("Select 2 or more processed documents to see shared themes, differences, and a summary.")

    selected_ids = _doc_picker("Choose documents to compare:", key="compare_docs", multiple=True)

    if selected_ids and len(selected_ids) >= 2:
        if st.button("Compare", type="primary"):
            docs_to_compare = {
                st.session_state.processed_docs[doc_id].file_name: st.session_state.processed_docs[doc_id].cleaned_text
                for doc_id in selected_ids
            }
            with st.spinner(f"Comparing {len(docs_to_compare)} documents..."):
                try:
                    result = compare_documents(docs_to_compare)

                    st.subheader("Overview")
                    st.write(result.get("overview", ""))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🔗 Shared Themes")
                        for theme in result.get("shared_themes", []):
                            st.markdown(f"- {theme}")
                    with col2:
                        st.subheader("⚡ Key Differences")
                        for diff in result.get("key_differences", []):
                            st.markdown(f"- {diff}")

                    st.subheader("Per-Document Summary")
                    per_doc = result.get("per_document_summary", {})
                    for name, summary in per_doc.items():
                        st.markdown(f"**{name}**: {summary}")

                    if result.get("recommendation"):
                        st.subheader("💡 Recommendation")
                        st.write(result["recommendation"])
                except Exception as e:
                    logger.exception("Comparison failed")
                    st.error(f"Comparison failed: {e}")
    elif selected_ids:
        st.info("Select at least 2 documents to compare.")
