"""Command-line interface for pen-tester-agent."""

import argparse

from .agent import agent_loop
from .providers import OllamaProvider
from .tools import default_registry
from . import ui



def main() -> None:
    """Parse arguments and start the agent loop."""
    parser = argparse.ArgumentParser(
        prog="pen-tester-agent",
        description="Penetration testing agent powered by local LLMs via Ollama",
    )
    parser.add_argument(
        "task", nargs="?", default=None,
        help="Task to perform (interactive if omitted)",
    )
    parser.add_argument(
        "--model", default="qwen2.5-coder:3b",
        help="Ollama model to use (default: qwen2.5-coder:3b)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=15,
        help="Max agent iterations (default: 15)",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=6000,
        help="Max context token budget (default: 6000)",
    )
    args = parser.parse_args()

    provider = OllamaProvider(model=args.model)
    registry = default_registry()

    if args.task:
        agent_loop(
            args.task, provider, registry,
            max_iterations=args.max_iterations,
            max_context_tokens=args.max_context_tokens,
        )
        return

    ui.show_banner()

    while True:
        choice = ui.show_menu()

        if choice == "quit":
            ui.show_goodbye()
            break

        if choice == "task":
            task = ui.prompt_task()
            if not task:
                ui.show_no_task()
                continue
            agent_loop(
                task, provider, registry,
                max_iterations=args.max_iterations,
                max_context_tokens=args.max_context_tokens,
            )

        elif choice == "report":
            ui.show_info("Start a task first — reports are generated from session history.")
