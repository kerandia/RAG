"""Rich terminal UI for the RAG agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rag_agent.agent import RAGAgent

console = Console()
logger = logging.getLogger(__name__)

_HELP_TEXT = """\
**Commands**
- Type a question to ask the RAG agent (full tool-use loop)
- `/rag <question>` — direct RAG (single-step, no tool loop)
- `/history`        — show conversation history (direct RAG path)
- `/clear`          — clear conversation history
- `/stats`          — show feedback rating statistics
- `/help`           — show this help message
- `exit` / `quit`   — quit the application
"""


def print_welcome(num_docs: int) -> None:
    """Print the application banner."""
    console.print(
        Panel(
            Text(
                "Logistics Feedback RAG Agent\n"
                f"Loaded {num_docs} feedback document(s)",
                justify="center",
                style="bold white",
            ),
            title="[bold cyan]RAG Agent[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print(Markdown(_HELP_TEXT))


def print_answer(answer: str, messages: list | None = None) -> None:
    """Print the agent answer, extracting tool-use steps from the message list."""
    console.print(
        Panel(
            Markdown(answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )
    )
    if messages:
        _print_tool_steps(messages)


def _print_tool_steps(messages: list) -> None:
    """Extract and display any tool calls from the LangGraph message list."""
    from langchain_core.messages import AIMessage, ToolMessage

    tool_calls = []
    tool_results: dict[str, str] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(tc)
        elif isinstance(msg, ToolMessage):
            tool_results[msg.tool_call_id] = str(msg.content)

    if not tool_calls:
        return

    table = Table(
        title="Tool calls",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Tool", style="cyan")
    table.add_column("Input")
    table.add_column("Output snippet")

    for i, tc in enumerate(tool_calls, 1):
        name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
        call_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
        result = tool_results.get(call_id, "")[:200]
        table.add_row(str(i), name, str(args)[:120], result)

    console.print(table)


def print_history(history: list[dict]) -> None:
    """Pretty-print the conversation history."""
    if not history:
        console.print("[dim]No conversation history.[/dim]")
        return
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        style = "bold blue" if role == "human" else "green"
        label = "You" if role == "human" else "Agent"
        console.print(f"[{style}]{label}:[/{style}] {content}")


def run_cli(agent: "RAGAgent", num_docs: int) -> None:
    """Start the interactive CLI loop.

    Args:
        agent: Initialised :class:`~rag_agent.agent.RAGAgent` instance.
        num_docs: Number of loaded documents (shown in the welcome banner).
    """
    print_welcome(num_docs)
    session_id = "cli"

    while True:
        try:
            raw = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not raw:
            continue

        if raw.lower() in ("exit", "quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if raw.lower() == "/help":
            console.print(Markdown(_HELP_TEXT))
            continue

        if raw.lower() == "/history":
            print_history(agent.get_history(session_id))
            continue

        if raw.lower() == "/clear":
            agent.clear_history(session_id)
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if raw.lower() == "/stats":
            with console.status("[bold yellow]Computing statistics...[/bold yellow]"):
                from rag_agent.tools import analyze_ratings
                stats_output = analyze_ratings.invoke({})
            console.print(
                Panel(
                    stats_output,
                    title="[bold yellow]Rating Statistics[/bold yellow]",
                    border_style="yellow",
                )
            )
            continue

        if raw.startswith("/rag "):
            question = raw[5:].strip()
            with console.status("[bold yellow]Retrieving and generating...[/bold yellow]"):
                answer = agent.direct_rag(question, session_id=session_id)
            print_answer(answer)
            continue

        # Default: full LangGraph agent with tool use
        with console.status("[bold yellow]Thinking...[/bold yellow]"):
            result = agent.chat(raw, session_id=session_id)

        print_answer(result["answer"], result.get("messages"))
