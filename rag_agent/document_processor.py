"""Load and process feedback documents into LangChain Document objects."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_feedback_documents(feedback_data_path: Path) -> list[Document]:
    """Load customer feedback from a JSON file.

    Each feedback entry is converted into a :class:`~langchain_core.documents.Document`
    with rich metadata (review_id, customer_id, rating, timestamp) so that
    downstream retrieval can filter on these fields.

    Args:
        feedback_data_path: Path to the ``feedback_data.json`` file.

    Returns:
        A list of :class:`~langchain_core.documents.Document` objects.

    Raises:
        FileNotFoundError: If the feedback file does not exist.
        ValueError: If the JSON structure is invalid.
    """
    if not feedback_data_path.exists():
        raise FileNotFoundError(f"Feedback file not found: {feedback_data_path}")

    with open(feedback_data_path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    entries = data.get("customer_feedback", [])
    if not isinstance(entries, list):
        raise ValueError("'customer_feedback' must be a JSON array.")

    documents: list[Document] = []
    for entry in entries:
        if not isinstance(entry, dict) or "feedback" not in entry:
            logger.warning("Skipping malformed entry: %s", entry)
            continue

        text = entry["feedback"].strip()
        if not text:
            continue

        metadata: dict[str, Any] = {
            "review_id": entry.get("review_id", ""),
            "customer_id": entry.get("customer_id", ""),
            "rating": entry.get("rating", None),
            "timestamp": entry.get("timestamp", ""),
            "source": str(feedback_data_path),
        }
        documents.append(Document(page_content=text, metadata=metadata))

    logger.info("Loaded %d feedback documents from %s", len(documents), feedback_data_path)
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents into smaller overlapping chunks while preserving metadata.

    Short feedback texts (< ``chunk_size`` tokens) are returned as-is.

    Args:
        documents: Input documents.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Overlap in characters between adjacent chunks.

    Returns:
        A list of chunked :class:`~langchain_core.documents.Document` objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunked = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunked))
    return chunked
