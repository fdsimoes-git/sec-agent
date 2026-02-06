import pytest

from sec_agent.tools import default_registry
from sec_agent.tools.base import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    """A fresh default tool registry for each test."""
    return default_registry()
