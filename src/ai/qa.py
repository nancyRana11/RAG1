"""
qa.py
-----
"Chat with PDF" and single-shot question-answering.

IMPORTANT DESIGN NOTE (read this before Phase 3):
This module uses "context stuffing" — it sends the ENTIRE (truncated)
document text as context on every question. That's simple, correct
for small-to-medium documents, and requires no extra infrastructure.

It does NOT scale well to very large documents or large document sets,
because:
  - You're bound by the model's context window
  - You pay for re-sending the same large context on every message
  - Relevant info can get "lost" in a huge context

Phase 3 replaces this with proper RAG: the document is chunked and
embedded once, and only the most RELEVANT chunks are retrieved and
sent per question — which scales to huge documents/libraries. The
public function signatures here (`ChatSession.ask`) are kept similar
on purpose so swapping the internals in Phase 3 doesn't require
rewriting the UI.
"""

from dataclasses import dataclass, field

from src.ai.llm_client import LLMClient
from src.utils.logger import logger

MAX_CONTEXT_CHARS = 50_000

QA_SYSTEM_PROMPT = """You are a careful assistant answering questions about a specific document.

Rules:
- Answer ONLY using information found in the provided document text.
- If the answer isn't in the document, say clearly: "I couldn't find that in the document."
- When possible, mention which part of the document supports your answer.
- Be concise and direct.
"""


@dataclass
class ChatMessage:
    role: str      # "user" or "assistant"
    content: str


@dataclass
class ChatSession:
    """
    Holds a running conversation about ONE document. Create one per
    document per user session (the Streamlit app keeps these in
    st.session_state, keyed by doc_id).
    """
    doc_id: str
    document_text: str
    history: list[ChatMessage] = field(default_factory=list)
    client: LLMClient = field(default_factory=LLMClient)

    def ask(self, question: str) -> str:
        """Asks a question in the context of the ongoing conversation and returns the answer."""
        context = self.document_text[:MAX_CONTEXT_CHARS]

        # Build the message list: document context goes in as part of the
        # first user turn's framing, then we replay prior Q&A turns so the
        # model has conversational memory (e.g. "what about the second one?").
        messages = []
        for msg in self.history:
            messages.append({"role": msg.role, "content": msg.content})

        user_turn = f"""DOCUMENT TEXT:
\"\"\"
{context}
\"\"\"

QUESTION: {question}"""
        messages.append({"role": "user", "content": user_turn})

        logger.info(f"[chat:{self.doc_id}] Q: {question[:80]}")
        answer = self.client.chat(messages, system=QA_SYSTEM_PROMPT, max_tokens=1000)

        # Store the ORIGINAL question (not the doc-stuffed version) in
        # history so replayed context doesn't balloon with duplicate
        # copies of the document on every turn.
        self.history.append(ChatMessage(role="user", content=question))
        self.history.append(ChatMessage(role="assistant", content=answer))

        return answer

    def reset(self):
        self.history = []


def ask_question(text: str, question: str, client: LLMClient | None = None) -> str:
    """
    Single-shot Q&A with no conversation memory — used for the
    "Question-answering over documents" feature where you just want
    one answer, not a running chat.
    """
    client = client or LLMClient()
    context = text[:MAX_CONTEXT_CHARS]

    prompt = f"""DOCUMENT TEXT:
\"\"\"
{context}
\"\"\"

QUESTION: {question}

Answer the question using ONLY the document text above."""

    logger.info(f"Single-shot QA: {question[:80]}")
    return client.complete(prompt, system=QA_SYSTEM_PROMPT, max_tokens=1000)
