"""
app.py
------
Streamlit entry point. Run with:

    streamlit run app.py

PHASE 1 SCOPE: upload a PDF, extract + clean text, generate an AI
summary, key points, and structured metadata. Later phases will add
more tabs/pages (Chat, Compare Documents, Semantic Search) to this
same app — the sidebar/session-state structure below is built to
extend cleanly.
"""

import streamlit as st

from config import validate_config, MAX_UPLOAD_MB
from src.utils.file_utils import save_uploaded_file, generate_doc_id
from src.document_processing.pipeline import process_document
from src.utils.logger import logger

st.set_page_config(page_title="Document Intelligence System", page_icon="📄", layout="wide")

# ------------------------------------------------------------------
# Session state: keeps processed documents around across reruns
# (Streamlit reruns the whole script on every interaction)
# ------------------------------------------------------------------
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = {}  # doc_id -> ProcessedDocument

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("📄 Doc Intelligence")
st.sidebar.caption("Phase 1: Core Document Processing")

config_problems = validate_config()
if config_problems:
    st.sidebar.error("Configuration problems:\n\n" + "\n".join(f"- {p}" for p in config_problems))
    st.sidebar.info("Copy `.env.example` to `.env` and fill in your API key, then restart the app.")

summary_style = st.sidebar.selectbox("Summary style", ["concise", "detailed", "executive"], index=0)
run_summary = st.sidebar.checkbox("Generate summary", value=True)
run_key_points = st.sidebar.checkbox("Generate key points", value=True)
run_metadata = st.sidebar.checkbox("Generate metadata", value=True)

st.sidebar.divider()
st.sidebar.caption(
    "Coming in later phases: OCR for scanned PDFs, chat with your "
    "document, multi-document comparison, semantic search, and an "
    "AI agent that routes between all of these automatically."
)

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
st.title("LLM-Powered Document Intelligence System")
st.write("Upload a PDF to extract its text and generate an AI summary, key points, and metadata.")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_UPLOAD_MB:
        st.error(f"File is {size_mb:.1f} MB, which exceeds the {MAX_UPLOAD_MB} MB limit.")
    else:
        doc_id = generate_doc_id(file_bytes)
        already_processed = doc_id in st.session_state.processed_docs

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**File:** {uploaded_file.name}  ·  **Size:** {size_mb:.2f} MB  ·  **ID:** `{doc_id}`")
        with col2:
            process_clicked = st.button(
                "🔄 Re-process" if already_processed else "▶ Process Document",
                type="primary",
                use_container_width=True,
            )

        if process_clicked or not already_processed:
            if config_problems:
                st.warning("Fix the configuration problems in the sidebar before processing.")
            else:
                with st.spinner("Saving file..."):
                    saved_path = save_uploaded_file(file_bytes, uploaded_file.name)

                with st.spinner("Extracting and analyzing document... this calls the LLM, please wait."):
                    try:
                        doc = process_document(
                            saved_path,
                            doc_id=doc_id,
                            summary_style=summary_style,
                            run_summary=run_summary,
                            run_key_points=run_key_points,
                            run_metadata=run_metadata,
                        )
                        st.session_state.processed_docs[doc_id] = doc
                        st.success("Document processed successfully.")
                    except Exception as e:
                        logger.exception("Processing failed")
                        st.error(f"Something went wrong while processing the document: {e}")

        # ------------------------------------------------------------
        # Render results, if we have them
        # ------------------------------------------------------------
        doc = st.session_state.processed_docs.get(doc_id)
        if doc:
            st.divider()

            if doc.extraction.scanned_page_ratio > 0.5:
                st.warning(
                    f"⚠️ {doc.extraction.scanned_page_ratio:.0%} of pages appear to have little/no "
                    "extractable text — this looks like a scanned document. OCR support is added "
                    "in Phase 2. Results below may be incomplete."
                )

            tabs = st.tabs(["📝 Summary", "🔑 Key Points", "🏷️ Metadata", "📊 Stats", "📄 Raw Text"])

            with tabs[0]:
                st.subheader(f"Summary ({summary_style})")
                st.write(doc.summary or "_Summary not generated (unchecked in sidebar)._")

            with tabs[1]:
                st.subheader("Key Points")
                if doc.key_points:
                    for i, point in enumerate(doc.key_points, start=1):
                        st.markdown(f"**{i}.** {point}")
                else:
                    st.write("_Key points not generated (unchecked in sidebar)._")

            with tabs[2]:
                st.subheader("Document Metadata (AI-extracted)")
                if doc.metadata:
                    st.json(doc.metadata)
                else:
                    st.write("_Metadata not generated (unchecked in sidebar)._")

                st.subheader("PDF File Metadata (from the file itself)")
                st.json(doc.extraction.pdf_metadata)

            with tabs[3]:
                st.subheader("Document Stats")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pages", doc.extraction.num_pages)
                c2.metric("Words", doc.stats["word_count"])
                c3.metric("Characters", doc.stats["char_count"])
                c4.metric("Est. tokens", doc.stats["estimated_tokens"])

            with tabs[4]:
                st.subheader("Cleaned Extracted Text")
                st.text_area("Full text", doc.cleaned_text, height=400)
                st.download_button(
                    "⬇ Download extracted text (.txt)",
                    data=doc.cleaned_text,
                    file_name=f"{doc.file_name}.extracted.txt",
                )
else:
    st.info("👆 Upload a PDF to get started.")
