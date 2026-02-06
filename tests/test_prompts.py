import json
import re

from sec_agent.prompts import ACTION_PATTERN, DONE_PATTERN, build_system_prompt
from sec_agent.tools import default_registry
from sec_agent.tools.base import ToolRegistry


class TestPatterns:
    def test_action_pattern_simple(self):
        text = 'ACTION: {"tool": "bash", "args": {"command": "ls"}}'
        m = ACTION_PATTERN.search(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "bash"
        assert parsed["args"]["command"] == "ls"

    def test_action_pattern_with_preceding_text(self):
        text = 'Let me check that.\n\nACTION: {"tool": "read_file", "args": {"path": "/etc/hosts"}}'
        m = ACTION_PATTERN.search(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["tool"] == "read_file"

    def test_action_pattern_no_match(self):
        text = "I will just explain things."
        m = ACTION_PATTERN.search(text)
        assert m is None

    def test_done_pattern(self):
        text = "DONE: completed the port scan"
        m = DONE_PATTERN.search(text)
        assert m is not None
        assert m.group(1) == "completed the port scan"

    def test_done_pattern_no_match(self):
        text = "Still working on it."
        m = DONE_PATTERN.search(text)
        assert m is None

    def test_action_pattern_nested_json(self):
        text = 'ACTION: {"tool": "http_request", "args": {"url": "https://example.com", "headers": {"Accept": "application/json"}}}'
        m = ACTION_PATTERN.search(text)
        assert m is not None
        parsed = json.loads(m.group(1))
        assert parsed["args"]["headers"]["Accept"] == "application/json"


class TestBuildSystemPrompt:
    def test_contains_tool_names(self):
        reg = default_registry()
        prompt = build_system_prompt(reg)
        assert "bash" in prompt
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "http_request" in prompt

    def test_contains_action_format(self):
        reg = default_registry()
        prompt = build_system_prompt(reg)
        assert "ACTION:" in prompt

    def test_contains_done_format(self):
        reg = default_registry()
        prompt = build_system_prompt(reg)
        assert "DONE:" in prompt

    def test_empty_registry(self):
        reg = ToolRegistry()
        prompt = build_system_prompt(reg)
        # Should still produce a valid prompt, just with no tool docs
        assert "ACTION:" in prompt
        assert "DONE:" in prompt
