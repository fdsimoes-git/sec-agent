"""Interactive tool-call approval for the agent loop."""

from . import ui


def ask_tool_approval(tool_name: str, args: dict) -> tuple[dict | None, bool]:
    """Request user approval before executing a tool call."""
    return ui.show_tool_approval_flow(tool_name, args)
