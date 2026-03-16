"""Persistent ChromaDB vector store backed by configurable embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from rag_agent.config import Settings

logger = logging.getLogger(__name__)


def _build_embeddings(settings: "Settings") -> Embeddings:
    """Instantiate the embedding model based on the configured provider.

    Args:
        settings: Application settings.

    Returns:
        A LangChain :class:`~langchain_core.embeddings.Embeddings` instance.
    """
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    # Default: Ollama
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def build_vector_store(settings: "Settings", documents: list[Document]) -> Chroma:
    """Create or update the persistent ChromaDB vector store.

    If the collection already contains documents it is reused without
    re-embedding, otherwise all ``documents`` are embedded and stored.

    Args:
        settings: Application settings.
        documents: Pre-chunked documents to index.

    Returns:
        A :class:`~langchain_chroma.Chroma` retriever-ready object.
    """
    embeddings = _build_embeddings(settings)
    persist_dir = str(settings.chroma_persist_dir)

    client = chromadb.PersistentClient(path=persist_dir)

    # Check if collection already populated
    try:
        collection = client.get_collection(settings.chroma_collection_name)
        existing_count = collection.count()
    except Exception:
        existing_count = 0

    vector_store = Chroma(
        client=client,
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
    )

    if existing_count == 0 and documents:
        logger.info("Indexing %d document chunks into ChromaDB...", len(documents))
        vector_store.add_documents(documents)
        logger.info("Indexing complete.")
    elif existing_count > 0:
        logger.info(
            "Reusing existing ChromaDB collection '%s' (%d docs).",
            settings.chroma_collection_name,
            existing_count,
        )

    return vector_store
