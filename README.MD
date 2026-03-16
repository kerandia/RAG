# Logistics Feedback RAG Agent

A production-quality **Retrieval-Augmented Generation (RAG) agent** for analysing
customer feedback from a logistics company.  Built with the latest LangChain
stack, ChromaDB persistent vector storage, hybrid retrieval and a beautiful Rich
terminal UI.

---

## Architecture

```
main.py                     # CLI entry point
rag_agent/
├── config.py               # Pydantic-based settings (env vars / .env file)
├── document_processor.py   # JSON loader + RecursiveCharacterTextSplitter
├── vector_store.py         # Persistent ChromaDB + configurable embeddings
├── retriever.py            # Hybrid BM25 + semantic retrieval with RRF fusion
├── tools.py                # LangChain tools: search, analytics, rating filter
├── agent.py                # LangChain ReAct agent + per-session memory
└── cli.py                  # Rich terminal UI (panels, tables, streaming)
tests/
└── test_rag_agent.py       # Pytest suite (mock LLM – no API keys needed)
```

### Key upgrades over the original script

| Feature | Original | New |
|---|---|---|
| Vector store | In-memory list | **Persistent ChromaDB** |
| Retrieval | Cosine similarity only | **Hybrid BM25 + semantic (RRF)** |
| Agent | None (Q&A loop) | **LangChain ReAct agent with tools** |
| Conversation | Stateless | **Per-session memory** |
| LLM provider | Ollama only | **Ollama or OpenAI (configurable)** |
| Document metadata | Not stored | **rating, review_id, timestamp** |
| UI | print() | **Rich panels + tables** |
| Config | Hardcoded | **Pydantic Settings + env vars** |

---

## Setup

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Start the Ollama server (local mode)

```bash
# Pull required models
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

# Start the server (runs on http://localhost:11434 by default)
ollama serve
```

### 3 – (Optional) Use OpenAI instead

```bash
export RAG_LLM_PROVIDER=openai
export RAG_OPENAI_API_KEY=sk-...
```

---

## Usage

```bash
# Run with defaults (Ollama, feedback_data.json in current directory)
python main.py

# Specify a different data file
python main.py --data /path/to/feedback.json

# Use OpenAI backend
python main.py --provider openai

# Increase retrieved documents
python main.py --top-k 8

# Enable verbose logging
python main.py --log-level INFO
```

### Interactive commands

| Command | Description |
|---|---|
| `<question>` | Ask the agent (uses ReAct loop with tools) |
| `/rag <question>` | Direct RAG (single-step, no tool loop) |
| `/stats` | Show rating statistics table |
| `/history` | Print conversation history |
| `/clear` | Clear conversation history |
| `/help` | Show help |
| `exit` / `quit` | Quit |

---

## Configuration

All settings can be overridden via environment variables prefixed with `RAG_`
or by creating a `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `RAG_LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `RAG_OLLAMA_EMBEDDING_MODEL` | `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` | Embedding model |
| `RAG_OLLAMA_LANGUAGE_MODEL` | `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` | Language model |
| `RAG_OPENAI_API_KEY` | _(empty)_ | OpenAI API key |
| `RAG_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `RAG_OPENAI_LANGUAGE_MODEL` | `gpt-4o-mini` | OpenAI language model |
| `RAG_CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `RAG_RETRIEVAL_TOP_K` | `5` | Documents retrieved per query |
| `RAG_RETRIEVAL_BM25_WEIGHT` | `0.4` | BM25 weight in hybrid retrieval |
| `RAG_RETRIEVAL_SEMANTIC_WEIGHT` | `0.6` | Semantic weight in hybrid retrieval |
| `RAG_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `RAG_FEEDBACK_DATA_PATH` | `./feedback_data.json` | Path to feedback JSON |

---

## Agent Tools

The ReAct agent can call the following tools autonomously:

| Tool | Description |
|---|---|
| `search_feedback` | Hybrid semantic + keyword search over feedback |
| `get_feedback_by_rating` | Filter feedback to a rating range (e.g. 1-2 for negatives) |
| `analyze_ratings` | Compute count, average, median, min/max and distribution |
| `summarize_all_feedback` | Return all feedback as a structured list |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests use mocked LLMs and embeddings – no running Ollama server or API key
required.

---

## Limitations

- The ChromaDB vector store must be rebuilt if `feedback_data.json` changes
  significantly (delete the `chroma_db/` directory to force a rebuild).
- Response quality depends on the quality of the chosen embedding and language models.
- The system is intentionally scoped to the loaded feedback corpus; it will not
  answer questions outside that context.
