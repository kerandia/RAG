# Logistics Feedback RAG Agent

A production-quality **Retrieval-Augmented Generation (RAG) agent** for analysing
customer feedback from a logistics company.  Built with the latest LangChain
stack, ChromaDB persistent vector storage, hybrid retrieval and a beautiful Rich
terminal UI.

---

## Architecture

```
main.py                     # CLI entry point
app.py                      # Gradio web UI (Hugging Face Spaces)
rag_agent/
├── config.py               # Pydantic-based settings (env vars / .env file)
├── document_processor.py   # JSON loader + RecursiveCharacterTextSplitter
├── vector_store.py         # Persistent ChromaDB + configurable embeddings
├── retriever.py            # Hybrid BM25 + semantic retrieval with RRF fusion
├── tools.py                # LangChain tools: search, analytics, rating filter
├── agent.py                # LangChain ReAct agent + per-session memory
└── cli.py                  # Rich terminal UI (panels, tables, streaming)
.github/
└── workflows/
    └── rag-analysis.yml    # GitHub Actions batch analysis
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
| LLM provider | Ollama only | **Ollama, OpenAI, or OpenRouter** |
| Document metadata | Not stored | **rating, review_id, timestamp** |
| UI | print() | **Rich panels + tables + Gradio web UI** |
| Config | Hardcoded | **Pydantic Settings + env vars** |

---

## Setup

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Configure your LLM provider

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

**Option A – OpenRouter (recommended, free tier available):**

```bash
RAG_LLM_PROVIDER=openrouter
RAG_OPENROUTER_API_KEY=sk-or-v1-...
RAG_OPENROUTER_LANGUAGE_MODEL=meta-llama/llama-3.2-3b-instruct:free
```

Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys) — no credit card required.
Embeddings run locally via `sentence-transformers/all-MiniLM-L6-v2` (no key needed).

Free models you can use (append `:free`):

| Model | Context | Notes |
|---|---|---|
| `meta-llama/llama-3.2-3b-instruct:free` | 128k | Fast, good for Q&A |
| `google/gemma-3-4b-it:free` | 32k | Google Gemma 3 |
| `mistralai/mistral-7b-instruct:free` | 32k | Mistral 7B |
| `deepseek/deepseek-r1:free` | 163k | Reasoning model |

Browse all free models at [openrouter.ai/models?q=:free](https://openrouter.ai/models?q=:free).

**Option B – Ollama (local, default):**

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
ollama serve
```

**Option C – OpenAI:**

```bash
RAG_LLM_PROVIDER=openai
RAG_OPENAI_API_KEY=sk-...
```

---

## Usage

```bash
# Run with defaults (Ollama, feedback_data.json in current directory)
python main.py

# Use OpenRouter (free)
python main.py --provider openrouter

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

## Hosting

### Option 1 – Hugging Face Spaces (free, public URL) ⭐ recommended

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces) → SDK: **Gradio**
2. Push this repository (or just `app.py`, `requirements.txt`, `feedback_data.json`, `rag_agent/`)
3. In your Space settings go to **Variables and secrets** and add:
   - `RAG_OPENROUTER_API_KEY` = your OpenRouter key
4. The Gradio web UI starts automatically — you get a permanent public URL

### Option 2 – GitHub Actions (batch analysis, no server required)

Run analysis on-demand or whenever `feedback_data.json` changes:

1. Add your OpenRouter key as a **GitHub Actions secret** named `OPENROUTER_API_KEY`
   (Settings → Secrets and variables → Actions → New repository secret)
2. Trigger the workflow manually from the **Actions** tab → *RAG Batch Analysis*,
   or it runs automatically on every push that modifies `feedback_data.json`
3. Download the JSON report from the workflow run's **Artifacts**

> GitHub Actions is a CI/CD runner, not a persistent service — it's perfect for
> scheduled or event-driven analysis jobs.

### Option 3 – GitHub Codespaces (development / demo)

1. Open this repo in Codespaces (Code → Codespaces → New codespace)
2. Run `pip install -r requirements.txt` in the terminal
3. Set `RAG_OPENROUTER_API_KEY` in the Codespace secrets (Settings → Codespaces)
4. `python main.py --provider openrouter` — runs in your browser

### Option 4 – Other free cloud platforms

| Platform | Free tier | How to deploy |
|---|---|---|
| [Render](https://render.com) | 750 h/month | Create a **Web Service**, set env vars, run `python app.py` |
| [Railway](https://railway.app) | $5 credit/month | Connect repo, set env vars |
| [Fly.io](https://fly.io) | 3 VMs free | `fly launch` in repo root |
| [Replit](https://replit.com) | Always-on for Deploying | Import repo, set Secrets, run `app.py` |

---

## Configuration

All settings can be overridden via environment variables prefixed with `RAG_`
or by creating a `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `RAG_LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or `openrouter` |
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `RAG_OLLAMA_EMBEDDING_MODEL` | `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` | Embedding model |
| `RAG_OLLAMA_LANGUAGE_MODEL` | `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` | Language model |
| `RAG_OPENAI_API_KEY` | _(empty)_ | OpenAI API key |
| `RAG_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `RAG_OPENAI_LANGUAGE_MODEL` | `gpt-4o-mini` | OpenAI language model |
| `RAG_OPENROUTER_API_KEY` | _(empty)_ | OpenRouter API key |
| `RAG_OPENROUTER_LANGUAGE_MODEL` | `meta-llama/llama-3.2-3b-instruct:free` | OpenRouter model |
| `RAG_HUGGINGFACE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embeddings (used with openrouter) |
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
