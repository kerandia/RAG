"""Tests for the RAG agent components.

These tests use mocked LLMs and embeddings so that they run without
requiring Ollama or an OpenAI API key.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FEEDBACK = {
    "customer_feedback": [
        {
            "review_id": "001",
            "customer_id": "C001",
            "feedback": "Delivery was quick and the package arrived undamaged.",
            "rating": 5,
            "timestamp": "2025-01-17T10:00:00Z",
        },
        {
            "review_id": "002",
            "customer_id": "C002",
            "feedback": "Package was damaged and the delivery was very late.",
            "rating": 1,
            "timestamp": "2025-01-18T12:00:00Z",
        },
        {
            "review_id": "003",
            "customer_id": "C003",
            "feedback": "Delivery arrived on time but box was slightly dented.",
            "rating": 3,
            "timestamp": "2025-01-19T14:00:00Z",
        },
    ]
}


@pytest.fixture()
def feedback_file(tmp_path: Path) -> Path:
    """Write sample feedback JSON to a temp file and return its path."""
    path = tmp_path / "feedback_data.json"
    path.write_text(json.dumps(SAMPLE_FEEDBACK), encoding="utf-8")
    return path


@pytest.fixture()
def settings(tmp_path: Path, feedback_file: Path):
    """Return Settings pointing at the temp feedback file."""
    from rag_agent.config import Settings

    return Settings(
        feedback_data_path=feedback_file,
        chroma_persist_dir=tmp_path / "chroma",
        llm_provider="ollama",
        retrieval_top_k=2,
    )


# ---------------------------------------------------------------------------
# document_processor tests
# ---------------------------------------------------------------------------


class TestDocumentProcessor:
    def test_load_feedback_documents(self, feedback_file: Path):
        from rag_agent.document_processor import load_feedback_documents

        docs = load_feedback_documents(feedback_file)
        assert len(docs) == 3
        assert docs[0].metadata["review_id"] == "001"
        assert docs[0].metadata["rating"] == 5

    def test_load_missing_file(self, tmp_path: Path):
        from rag_agent.document_processor import load_feedback_documents

        with pytest.raises(FileNotFoundError):
            load_feedback_documents(tmp_path / "nonexistent.json")

    def test_load_skips_malformed_entries(self, tmp_path: Path):
        from rag_agent.document_processor import load_feedback_documents

        bad_data = {
            "customer_feedback": [
                {"review_id": "001", "feedback": "Good delivery.", "rating": 5},
                {"review_id": "002"},  # missing 'feedback' key
                "just a string",  # not a dict
            ]
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_data), encoding="utf-8")
        docs = load_feedback_documents(path)
        assert len(docs) == 1  # only the valid entry

    def test_chunk_documents(self, feedback_file: Path):
        from rag_agent.document_processor import chunk_documents, load_feedback_documents

        docs = load_feedback_documents(feedback_file)
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
        # Metadata must be preserved in all chunks
        for chunk in chunks:
            assert "review_id" in chunk.metadata


# ---------------------------------------------------------------------------
# config tests
# ---------------------------------------------------------------------------


class TestSettings:
    def test_default_provider_is_ollama(self):
        from rag_agent.config import Settings

        s = Settings()
        assert s.llm_provider == "ollama"

    def test_custom_top_k(self, feedback_file: Path, tmp_path: Path):
        from rag_agent.config import Settings

        s = Settings(feedback_data_path=feedback_file, chroma_persist_dir=tmp_path, retrieval_top_k=10)
        assert s.retrieval_top_k == 10

    def test_hybrid_weights_defaults(self):
        from rag_agent.config import Settings

        s = Settings()
        assert s.retrieval_bm25_weight + s.retrieval_semantic_weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# retriever tests (with mock vector store)
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    def _make_documents(self):
        from langchain_core.documents import Document

        return [
            Document(
                page_content="Delivery was quick and undamaged.",
                metadata={"review_id": "001", "rating": 5},
            ),
            Document(
                page_content="Package was damaged and late.",
                metadata={"review_id": "002", "rating": 1},
            ),
            Document(
                page_content="Delivery on time but box slightly dented.",
                metadata={"review_id": "003", "rating": 3},
            ),
        ]

    def _make_mock_vector_store(self, docs):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = docs[:2]
        return mock_vs

    def test_retrieve_returns_top_k(self, settings):
        from rag_agent.retriever import HybridRetriever

        docs = self._make_documents()
        mock_vs = self._make_mock_vector_store(docs)
        retriever = HybridRetriever(mock_vs, docs, settings)
        results = retriever.retrieve("delivery time", top_k=2)
        assert len(results) <= 2

    def test_retrieve_rating_filter(self, settings):
        from rag_agent.retriever import HybridRetriever

        docs = self._make_documents()
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [docs[0]]  # only rating 5
        retriever = HybridRetriever(mock_vs, docs, settings)
        results = retriever.retrieve("delivery", min_rating=4.0)
        for doc in results:
            rating = doc.metadata.get("rating")
            if rating is not None:
                assert rating >= 4.0

    def test_rrf_deduplicates(self, settings):
        from rag_agent.retriever import HybridRetriever

        docs = self._make_documents()
        mock_vs = MagicMock()
        # Return all three docs from semantic search
        mock_vs.similarity_search.return_value = docs
        retriever = HybridRetriever(mock_vs, docs, settings)
        results = retriever.retrieve("package")
        # No duplicates
        ids = [d.metadata["review_id"] for d in results]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# tools tests
# ---------------------------------------------------------------------------


class TestTools:
    def _setup(self, docs):
        from rag_agent.tools import init_tools

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = docs
        init_tools(mock_retriever, docs)

    def _make_docs(self):
        from langchain_core.documents import Document

        return [
            Document(
                page_content="Delivery was quick.",
                metadata={"review_id": "001", "rating": 5},
            ),
            Document(
                page_content="Package was damaged.",
                metadata={"review_id": "002", "rating": 1},
            ),
        ]

    def test_search_feedback_returns_results(self):
        from rag_agent.tools import search_feedback

        docs = self._make_docs()
        self._setup(docs)
        result = search_feedback.invoke({"query": "delivery"})
        assert "001" in result
        assert "Rating 5/5" in result

    def test_analyze_ratings_returns_stats(self):
        from rag_agent.tools import analyze_ratings, init_tools

        docs = self._make_docs()
        mock_retriever = MagicMock()
        init_tools(mock_retriever, docs)

        result = analyze_ratings.invoke({})
        data = json.loads(result)
        assert data["count"] == 2
        assert data["average"] == 3.0

    def test_get_feedback_by_rating_filters(self):
        from rag_agent.tools import get_feedback_by_rating, init_tools

        docs = self._make_docs()
        mock_retriever = MagicMock()
        init_tools(mock_retriever, docs)

        result = get_feedback_by_rating.invoke({"min_rating": 4.0, "max_rating": 5.0})
        assert "001" in result
        assert "002" not in result

    def test_summarize_all_feedback(self):
        from rag_agent.tools import init_tools, summarize_all_feedback

        docs = self._make_docs()
        mock_retriever = MagicMock()
        init_tools(mock_retriever, docs)

        result = summarize_all_feedback.invoke({})
        assert "001" in result
        assert "002" in result


# ---------------------------------------------------------------------------
# agent tests (with mocked LLM)
# ---------------------------------------------------------------------------


class TestRAGAgent:
    def _make_docs(self):
        from langchain_core.documents import Document

        return [
            Document(
                page_content="Delivery was quick and undamaged.",
                metadata={"review_id": "001", "rating": 5},
            ),
        ]

    def test_direct_rag_returns_string(self, settings):
        from langchain_core.messages import AIMessage
        from rag_agent.agent import RAGAgent

        docs = self._make_docs()
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = docs

        with patch("rag_agent.agent._build_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="The delivery was fast.")
            mock_llm_fn.return_value = mock_llm

            agent = RAGAgent(settings=settings, vector_store=mock_vs, documents=docs)
            answer = agent.direct_rag("How was the delivery?")

        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_history_persists(self, settings):
        from langchain_core.messages import AIMessage
        from rag_agent.agent import RAGAgent

        docs = self._make_docs()
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = docs

        with patch("rag_agent.agent._build_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="Answer 1")
            mock_llm_fn.return_value = mock_llm

            agent = RAGAgent(settings=settings, vector_store=mock_vs, documents=docs)
            agent.direct_rag("First question", session_id="test")
            history = agent.get_history("test")

        assert len(history) == 2  # human + AI

    def test_clear_history(self, settings):
        from langchain_core.messages import AIMessage
        from rag_agent.agent import RAGAgent

        docs = self._make_docs()
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = docs

        with patch("rag_agent.agent._build_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="Some answer")
            mock_llm_fn.return_value = mock_llm

            agent = RAGAgent(settings=settings, vector_store=mock_vs, documents=docs)
            agent.direct_rag("First question", session_id="test")
            agent.clear_history("test")
            history = agent.get_history("test")

        assert len(history) == 0
