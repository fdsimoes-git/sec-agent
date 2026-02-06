from .base import Tool, ToolRegistry, ToolResult
from .bash_tool import BashTool
from .http_request import HttpRequestTool
from .read_file import ReadFileTool
from .write_file import WriteFileTool

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "BashTool",
    "HttpRequestTool",
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
    return registry
