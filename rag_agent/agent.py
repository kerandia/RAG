"""LangChain 1.x LangGraph agent with conversation memory and tool use."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.checkpoint.memory import MemorySaver

from rag_agent.tools import get_all_tools, init_tools

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from rag_agent.config import Settings
    from rag_agent.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a logistics customer-feedback analyst powered by Retrieval-Augmented Generation (RAG).

You have access to tools that let you search, filter, and analyse customer feedback data.
Always ground your answers in the retrieved data. Do not hallucinate facts.

When answering:
1. Use the tools to retrieve relevant feedback before drawing conclusions.
2. Cite the review IDs you used (e.g. "According to review #001 ...").
3. Be concise yet thorough. Use bullet points where appropriate.
4. If the retrieved context does not contain enough information, say so clearly.
"""


def _build_llm(settings: "Settings") -> "BaseChatModel":
    """Construct the configured chat model."""
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_language_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            streaming=settings.streaming,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_language_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )


class RAGAgent:
    """High-level RAG agent interface.

    Wraps a LangChain 1.x LangGraph agent backed by a hybrid retriever and
    persistent ChromaDB vector store. Maintains per-session conversation history
    via a LangGraph ``MemorySaver`` checkpointer.

    Args:
        settings: Application settings.
        vector_store: Populated Chroma vector store.
        documents: All indexed documents (used by BM25 and analytics tools).
    """

    def __init__(
        self,
        settings: "Settings",
        vector_store: "Chroma",
        documents: list["Document"],
    ) -> None:
        self._settings = settings
        self._documents = documents

        # Build the hybrid retriever
        from rag_agent.retriever import HybridRetriever

        self._retriever: HybridRetriever = HybridRetriever(
            vector_store=vector_store,
            documents=documents,
            settings=settings,
        )

        # Register tools (module-level singletons)
        init_tools(self._retriever, documents)

        # Per-session chat histories (for the direct_rag path)
        self._histories: dict[str, ChatMessageHistory] = {}

        # Build LLM  
        self._llm = _build_llm(settings)
        tools = get_all_tools()

        # Create LangGraph agent with in-memory checkpointer for conversation memory
        checkpointer = MemorySaver()
        self._agent = create_agent(
            self._llm,
            tools=tools,
            system_prompt=_SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str, session_id: str = "default") -> dict[str, Any]:
        """Send *user_message* to the agent and return the response.

        The LangGraph agent automatically uses the checkpointer to persist
        conversation history per ``session_id``.

        Args:
            user_message: The user's question.
            session_id: Identifier for the conversation session (thread).

        Returns:
            A dict with ``"answer"`` (str) and ``"messages"`` (full message list).
        """
        config = {"configurable": {"thread_id": session_id}}
        state = self._agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        messages = state.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if hasattr(m, "content") and m.type == "ai"),
            None,
        )
        answer = last_ai.content if last_ai else ""
        return {"answer": answer, "messages": messages}

    def direct_rag(self, user_message: str, session_id: str = "default") -> str:
        """Simpler RAG path: retrieve context then answer in a single LLM call.

        Does not use the tool loop. Useful for quick, focused answers.

        Args:
            user_message: The user's question.
            session_id: Identifier for the conversation session.

        Returns:
            The LLM's answer string.
        """
        docs = self._retriever.retrieve(user_message)
        context = "\n".join(
            f"[{d.metadata.get('review_id', '?')}] "
            f"Rating {d.metadata.get('rating', 'N/A')}/5: {d.page_content}"
            for d in docs
        )

        history = self._get_history(session_id)

        system_content = (
            "You are a logistics customer-feedback analyst. "
            "Use ONLY the following retrieved feedback to answer the question. "
            "Cite review IDs. If the context is insufficient, say so.\n\n"
            f"Retrieved context:\n{context}"
        )
        messages: list = [("system", system_content)]
        messages.extend((m.type, m.content) for m in history.messages)
        messages.append(("human", user_message))

        response = self._llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        history.add_user_message(user_message)
        history.add_ai_message(answer)
        return answer

    def clear_history(self, session_id: str = "default") -> None:
        """Clear the conversation history for *session_id*.

        Note: this clears the direct_rag LCEL history; the LangGraph agent
        history is scoped to the checkpointer thread.
        """
        if session_id in self._histories:
            self._histories[session_id].clear()

    def get_history(self, session_id: str = "default") -> list[dict[str, str]]:
        """Return the direct_rag conversation history as role/content dicts."""
        history = self._get_history(session_id)
        return [{"role": m.type, "content": m.content} for m in history.messages]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = ChatMessageHistory()
        return self._histories[session_id]
