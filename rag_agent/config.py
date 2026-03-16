"""Configuration and settings management using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider
    llm_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="LLM backend provider: 'ollama' (local) or 'openai'.",
    )

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama server.",
    )
    ollama_embedding_model: str = Field(
        default="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
        description="Ollama embedding model name.",
    )
    ollama_language_model: str = Field(
        default="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
        description="Ollama language model name.",
    )

    # OpenAI settings (used when llm_provider='openai')
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (set RAG_OPENAI_API_KEY env var).",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )
    openai_language_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI language model name.",
    )

    # Vector store
    chroma_persist_dir: Path = Field(
        default=Path("./chroma_db"),
        description="Directory for ChromaDB persistence.",
    )
    chroma_collection_name: str = Field(
        default="logistics_feedback",
        description="Name of the ChromaDB collection.",
    )

    # Retrieval
    retrieval_top_k: int = Field(
        default=5,
        description="Number of documents to retrieve.",
    )
    retrieval_bm25_weight: float = Field(
        default=0.4,
        description="Weight for BM25 scores in hybrid retrieval (0-1).",
    )
    retrieval_semantic_weight: float = Field(
        default=0.6,
        description="Weight for semantic scores in hybrid retrieval (0-1).",
    )

    # Data
    feedback_data_path: Path = Field(
        default=Path("./feedback_data.json"),
        description="Path to the feedback JSON data file.",
    )

    # Agent
    agent_max_iterations: int = Field(
        default=8,
        description="Maximum reasoning steps for the ReAct agent.",
    )
    streaming: bool = Field(
        default=True,
        description="Whether to stream LLM output token-by-token.",
    )
    temperature: float = Field(
        default=0.1,
        description="LLM sampling temperature (lower = more factual).",
    )
