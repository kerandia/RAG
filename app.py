#!/usr/bin/env python3
"""Gradio web interface for the Logistics Feedback RAG Agent.

Deploy to Hugging Face Spaces (free tier):
  1. Create a new Space at https://huggingface.co/spaces
  2. Set SDK to "Gradio"
  3. Push this repo (or copy app.py + requirements.txt + feedback_data.json)
  4. Add your OPENROUTER_API_KEY in Space Secrets (Settings > Variables and secrets)

The Space will install requirements.txt and launch this file automatically.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Lazy initialisation – agent is loaded once on first use
# ---------------------------------------------------------------------------
_agent = None


def _get_agent():
    """Initialise the RAG agent on first call (cached)."""
    global _agent
    if _agent is not None:
        return _agent

    from rag_agent.config import Settings
    from rag_agent.document_processor import chunk_documents, load_feedback_documents
    from rag_agent.vector_store import build_vector_store
    from rag_agent.agent import RAGAgent

    settings = Settings(
        llm_provider=os.environ.get("RAG_LLM_PROVIDER", "openrouter"),
        openrouter_api_key=os.environ.get("RAG_OPENROUTER_API_KEY", ""),
        openrouter_language_model=os.environ.get(
            "RAG_OPENROUTER_LANGUAGE_MODEL",
            "meta-llama/llama-3.2-3b-instruct:free",
        ),
        feedback_data_path=Path(
            os.environ.get("RAG_FEEDBACK_DATA_PATH", "./feedback_data.json")
        ),
        chroma_persist_dir=Path(
            os.environ.get("RAG_CHROMA_PERSIST_DIR", "/tmp/chroma_db")
        ),
    )

    raw_docs = load_feedback_documents(settings.feedback_data_path)
    documents = chunk_documents(raw_docs)
    vector_store = build_vector_store(settings, documents)
    _agent = RAGAgent(settings=settings, vector_store=vector_store, documents=documents)
    return _agent


# ---------------------------------------------------------------------------
# Gradio chat handler
# ---------------------------------------------------------------------------


def chat(message: str, history: list[list[str]]) -> str:
    """Process a chat message and return the agent's response."""
    if not message.strip():
        return ""
    try:
        agent = _get_agent()
        result = agent.chat(message, session_id="gradio")
        return result["answer"]
    except Exception as exc:
        logging.exception("Agent error")
        return f"⚠️ Error: {exc}"


def direct_rag(message: str, history: list[list[str]]) -> str:
    """Single-step RAG answer without tool loop."""
    if not message.strip():
        return ""
    try:
        agent = _get_agent()
        return agent.direct_rag(message, session_id="gradio-direct")
    except Exception as exc:
        logging.exception("Agent error")
        return f"⚠️ Error: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_DESCRIPTION = """
## 🚚 Logistics Feedback RAG Agent

Ask questions about customer feedback data. The agent uses **hybrid retrieval**
(BM25 + semantic search with RRF fusion) backed by a persistent **ChromaDB** vector
store and a free **OpenRouter** LLM.

**Example questions:**
- What are the most common complaints?
- Summarise the positive feedback
- How many reviews gave a rating below 3?
- What do customers say about delivery times?

> Powered by LangChain 1.x · ChromaDB · OpenRouter (free LLM) · sentence-transformers
"""

with gr.Blocks(title="Logistics RAG Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_DESCRIPTION)

    with gr.Tab("🤖 Agent (with tools)"):
        gr.ChatInterface(
            fn=chat,
            chatbot=gr.Chatbot(height=420),
            textbox=gr.Textbox(
                placeholder="Ask about customer feedback...",
                container=False,
                scale=7,
            ),
            title="",
            examples=[
                "What are the most common complaints?",
                "Summarise positive reviews (rating 4-5)",
                "Show me the rating distribution",
                "What do customers say about packaging?",
            ],
            cache_examples=False,
        )

    with gr.Tab("⚡ Direct RAG (fast)"):
        gr.ChatInterface(
            fn=direct_rag,
            chatbot=gr.Chatbot(height=420),
            textbox=gr.Textbox(
                placeholder="Ask about customer feedback (single-step, no tool loop)...",
                container=False,
                scale=7,
            ),
            title="",
            cache_examples=False,
        )

if __name__ == "__main__":
    demo.launch()
