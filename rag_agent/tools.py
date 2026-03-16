"""LangChain tools exposed to the ReAct agent."""

from __future__ import annotations

import json
import logging
import statistics
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from rag_agent.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state – set once during agent initialisation
# ---------------------------------------------------------------------------
_retriever: "HybridRetriever | None" = None
_all_documents: "list[Document] | None" = None


def init_tools(retriever: "HybridRetriever", documents: "list[Document]") -> None:
    """Bind the shared retriever and document list to all tools.

    Must be called before any tool is invoked.
    """
    global _retriever, _all_documents
    _retriever = retriever
    _all_documents = documents


def _require_retriever() -> "HybridRetriever":
    if _retriever is None:
        raise RuntimeError("Tools have not been initialised – call init_tools() first.")
    return _retriever


def _require_documents() -> "list[Document]":
    if _all_documents is None:
        raise RuntimeError("Tools have not been initialised – call init_tools() first.")
    return _all_documents


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool
def search_feedback(query: str) -> str:
    """Search customer feedback documents using hybrid semantic + keyword retrieval.

    Use this tool to find the most relevant feedback entries for a given topic or question.

    Args:
        query: Natural-language question or topic to search for.

    Returns:
        A formatted string with matching feedback entries, their ratings and IDs.
    """
    retriever = _require_retriever()
    docs = retriever.retrieve(query)
    if not docs:
        return "No relevant feedback found."
    lines = []
    for doc in docs:
        meta = doc.metadata
        rating = meta.get("rating", "N/A")
        review_id = meta.get("review_id", "?")
        lines.append(f"[{review_id}] Rating {rating}/5: {doc.page_content}")
    return "\n".join(lines)


@tool
def get_feedback_by_rating(min_rating: float = 1.0, max_rating: float = 5.0) -> str:
    """Retrieve feedback entries filtered to a specific rating range.

    Useful for finding only negative (1-2), mixed (3), or positive (4-5) reviews.

    Args:
        min_rating: Minimum rating score (1-5), inclusive.
        max_rating: Maximum rating score (1-5), inclusive.

    Returns:
        A formatted string of matching feedback entries.
    """
    docs = _require_documents()
    filtered = [
        d for d in docs
        if d.metadata.get("rating") is not None
        and min_rating <= d.metadata["rating"] <= max_rating
    ]
    if not filtered:
        return f"No feedback found with rating between {min_rating} and {max_rating}."
    lines = []
    for doc in filtered:
        meta = doc.metadata
        lines.append(
            f"[{meta.get('review_id', '?')}] Rating {meta.get('rating')}/5: {doc.page_content}"
        )
    return "\n".join(lines)


@tool
def analyze_ratings() -> str:
    """Compute descriptive statistics over all customer ratings.

    Returns count, average, median, min, max and a rating distribution breakdown.

    Returns:
        A JSON-formatted string with rating statistics.
    """
    docs = _require_documents()
    ratings = [d.metadata["rating"] for d in docs if d.metadata.get("rating") is not None]
    if not ratings:
        return "No rating data available."

    distribution: dict[int, int] = {}
    for r in ratings:
        key = int(round(r))
        distribution[key] = distribution.get(key, 0) + 1

    stats: dict[str, Any] = {
        "count": len(ratings),
        "average": round(statistics.mean(ratings), 2),
        "median": statistics.median(ratings),
        "min": min(ratings),
        "max": max(ratings),
        "distribution": {f"{k} stars": v for k, v in sorted(distribution.items())},
    }
    return json.dumps(stats, indent=2)


@tool
def summarize_all_feedback() -> str:
    """Return a complete listing of all loaded feedback entries.

    Useful as a starting point before synthesising a high-level summary.

    Returns:
        All feedback entries formatted as a numbered list.
    """
    docs = _require_documents()
    if not docs:
        return "No feedback data loaded."
    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        lines.append(
            f"{i}. [{meta.get('review_id', '?')}] "
            f"Rating {meta.get('rating', 'N/A')}/5 | "
            f"{meta.get('timestamp', '')} | "
            f"{doc.page_content}"
        )
    return "\n".join(lines)


def get_all_tools() -> list:
    """Return all agent tools as a list."""
    return [
        search_feedback,
        get_feedback_by_rating,
        analyze_ratings,
        summarize_all_feedback,
    ]
