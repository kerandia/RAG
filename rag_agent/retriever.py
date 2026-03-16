"""Hybrid BM25 + semantic retrieval with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from rag_agent.config import Settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine BM25 keyword search with dense semantic retrieval via RRF.

    Reciprocal Rank Fusion merges results from both ranking lists without
    requiring score normalisation.

    Args:
        vector_store: Populated :class:`~langchain_chroma.Chroma` instance.
        documents: All indexed documents (used to build the BM25 index).
        settings: Application settings.
    """

    _RRF_K = 60  # RRF constant; higher values reduce the impact of top ranks

    def __init__(
        self,
        vector_store: "Chroma",
        documents: list[Document],
        settings: "Settings",
    ) -> None:
        self._vector_store = vector_store
        self._documents = documents
        self._settings = settings
        self._bm25, self._bm25_docs = self._build_bm25_index(documents)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase whitespace tokenisation."""
        return text.lower().split()

    def _build_bm25_index(
        self, documents: list[Document]
    ) -> tuple[BM25Okapi, list[Document]]:
        corpus = [self._tokenize(doc.page_content) for doc in documents]
        return BM25Okapi(corpus), documents

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_rating: float | None = None,
        max_rating: float | None = None,
    ) -> list[Document]:
        """Retrieve the most relevant documents for *query* using hybrid search.

        Args:
            query: The user question.
            top_k: Number of results to return (defaults to ``settings.retrieval_top_k``).
            min_rating: If provided, filter out docs with rating < min_rating.
            max_rating: If provided, filter out docs with rating > max_rating.

        Returns:
            A list of :class:`~langchain_core.documents.Document` objects ordered
            by relevance (highest first).
        """
        k = top_k or self._settings.retrieval_top_k

        # --- Semantic search ---
        semantic_filter = self._build_chroma_filter(min_rating, max_rating)
        semantic_results: list[Document] = self._vector_store.similarity_search(
            query, k=k * 2, filter=semantic_filter
        )

        # --- BM25 search ---
        bm25_scores = self._bm25.get_scores(self._tokenize(query))
        bm25_ranked = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )
        bm25_docs = [self._bm25_docs[i] for i, _ in bm25_ranked[: k * 2]]

        # Apply rating filter to BM25 results
        if min_rating is not None or max_rating is not None:
            bm25_docs = [
                d for d in bm25_docs if self._rating_passes(d, min_rating, max_rating)
            ]

        # --- Reciprocal Rank Fusion ---
        fused = self._rrf(semantic_results, bm25_docs)
        return fused[:k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rating_passes(
        doc: Document,
        min_rating: float | None,
        max_rating: float | None,
    ) -> bool:
        rating = doc.metadata.get("rating")
        if rating is None:
            return True
        if min_rating is not None and rating < min_rating:
            return False
        if max_rating is not None and rating > max_rating:
            return False
        return True

    @staticmethod
    def _build_chroma_filter(
        min_rating: float | None,
        max_rating: float | None,
    ) -> dict | None:
        """Construct a ChromaDB metadata filter dict for rating range."""
        conditions: list[dict] = []
        if min_rating is not None:
            conditions.append({"rating": {"$gte": min_rating}})
        if max_rating is not None:
            conditions.append({"rating": {"$lte": max_rating}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _rrf(
        self,
        semantic: list[Document],
        bm25: list[Document],
    ) -> list[Document]:
        """Merge two ranked lists via Reciprocal Rank Fusion.

        Returns de-duplicated documents ordered by fused RRF score.
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        def _doc_key(doc: Document) -> str:
            return doc.metadata.get("review_id") or doc.page_content[:80]

        for rank, doc in enumerate(semantic):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + (
                self._settings.retrieval_semantic_weight / (self._RRF_K + rank + 1)
            )
            doc_map[key] = doc

        for rank, doc in enumerate(bm25):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + (
                self._settings.retrieval_bm25_weight / (self._RRF_K + rank + 1)
            )
            doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]
