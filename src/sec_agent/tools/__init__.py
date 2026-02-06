"""Built-in tools and registry factory."""

from .base import Tool, ToolRegistry, ToolResult
from .bash_tool import BashTool
from .done import DoneTool
from .http_request import HttpRequestTool
from .math_tool import MathTool
from .read_file import ReadFileTool
from .write_file import WriteFileTool

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "BashTool",
    "DoneTool",
    "HttpRequestTool",
    "MathTool",
    "ReadFileTool",
    "WriteFileTool",
    "default_registry",
]


def default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with all built-in tools."""
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(HttpRequestTool())
    registry.register(MathTool())
    registry.register(DoneTool())
    return registry
