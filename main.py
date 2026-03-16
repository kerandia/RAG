#!/usr/bin/env python3
"""Entry point for the Logistics Feedback RAG Agent."""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console

console = Console()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Logistics Feedback RAG Agent – ask questions about customer reviews.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default=None,
        help="LLM backend (overrides RAG_LLM_PROVIDER env var).",
    )
    parser.add_argument(
        "--data",
        default=None,
        metavar="PATH",
        help="Path to feedback_data.json (overrides RAG_FEEDBACK_DATA_PATH env var).",
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        metavar="DIR",
        help="Directory for ChromaDB persistence (overrides RAG_CHROMA_PERSIST_DIR).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of documents to retrieve per query.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build settings (CLI overrides > env vars > defaults)
    from rag_agent.config import Settings

    kwargs: dict = {}
    if args.provider:
        kwargs["llm_provider"] = args.provider
    if args.data:
        kwargs["feedback_data_path"] = args.data
    if args.chroma_dir:
        kwargs["chroma_persist_dir"] = args.chroma_dir
    if args.top_k:
        kwargs["retrieval_top_k"] = args.top_k

    settings = Settings(**kwargs)

    # Load and process documents
    from rag_agent.document_processor import chunk_documents, load_feedback_documents

    with console.status("[bold cyan]Loading feedback data...[/bold cyan]"):
        try:
            raw_docs = load_feedback_documents(settings.feedback_data_path)
        except FileNotFoundError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)
        documents = chunk_documents(raw_docs)

    # Build or load vector store
    from rag_agent.vector_store import build_vector_store

    with console.status("[bold cyan]Setting up vector store...[/bold cyan]"):
        vector_store = build_vector_store(settings, documents)

    # Initialise agent
    from rag_agent.agent import RAGAgent

    with console.status("[bold cyan]Initialising RAG agent...[/bold cyan]"):
        agent = RAGAgent(
            settings=settings,
            vector_store=vector_store,
            documents=documents,
        )

    # Launch interactive CLI
    from rag_agent.cli import run_cli

    run_cli(agent, num_docs=len(raw_docs))


if __name__ == "__main__":
    main()
