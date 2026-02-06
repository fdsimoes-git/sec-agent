import json
import pytest

from sec_agent.agent import agent_loop
from sec_agent.providers.base import ModelProvider
from sec_agent.tools import default_registry


class FakeProvider(ModelProvider):
    """A fake provider that returns pre-scripted responses."""

    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    def chat(self, messages: list[dict]) -> str:
        return next(self._responses)


class TestAgentLoop:
    def test_done_on_first_response(self, monkeypatch):
        """Agent exits cleanly when LLM responds with DONE immediately."""
        provider = FakeProvider(["DONE: nothing to do"])
        registry = default_registry()
        # Empty input on follow-up prompt means quit
        monkeypatch.setattr("builtins.input", lambda _: "")
        agent_loop("do something", provider, registry, max_iterations=5)

    def test_done_with_followup(self, monkeypatch, capsys):
        """Agent handles follow-up after DONE."""
        provider = FakeProvider([
            "DONE: listed the files",
            "DONE: also counted them",
        ])
        registry = default_registry()
        inputs = iter(["count them too", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("list files", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "listed the files" in out
        assert "also counted them" in out

    def test_tool_call_approved(self, monkeypatch, capsys):
        """Agent executes a tool when user approves."""
        action = json.dumps({"tool": "bash", "args": {"command": "echo test123"}})
        provider = FakeProvider([
            f"I'll run echo.\n\nACTION: {action}",
            "DONE: ran the command",
        ])
        registry = default_registry()
        # First input: approve the tool call. Second: quit after DONE.
        inputs = iter(["y", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("echo test", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "test123" in out

    def test_tool_call_rejected(self, monkeypatch, capsys):
        """Agent handles user rejecting a tool call."""
        action = json.dumps({"tool": "bash", "args": {"command": "rm -rf /"}})
        provider = FakeProvider([
            f"ACTION: {action}",
            "DONE: understood, cancelled",
        ])
        registry = default_registry()
        inputs = iter(["n", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("do something dangerous", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "cancelled" in out.lower()

    def test_unknown_tool(self, monkeypatch, capsys):
        """Agent handles LLM requesting a non-existent tool."""
        action = json.dumps({"tool": "nonexistent", "args": {}})
        provider = FakeProvider([
            f"ACTION: {action}",
            "DONE: ok nevermind",
        ])
        registry = default_registry()
        monkeypatch.setattr("builtins.input", lambda _: "")
        agent_loop("do something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "Unknown tool" in out

    def test_invalid_json_action(self, monkeypatch, capsys):
        """Agent handles malformed JSON in ACTION and recovers."""
        provider = FakeProvider([
            'ACTION: {not: valid json}',
            "DONE: gave up",
        ])
        registry = default_registry()
        monkeypatch.setattr("builtins.input", lambda _: "")
        agent_loop("do something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        # Agent should recover: warn about bad JSON, then get DONE on next turn
        assert "could not parse" in out.lower()
        assert "gave up" in out.lower()

    def test_plain_text_response(self, monkeypatch, capsys):
        """Agent asks user for input when LLM doesn't use a tool."""
        provider = FakeProvider([
            "What IP address should I scan?",
            "DONE: ok, quitting",
        ])
        registry = default_registry()
        inputs = iter(["192.168.1.1", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("scan something", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "What IP" in out

    def test_max_iterations(self, monkeypatch, capsys):
        """Agent stops at max iterations."""
        # Always returns plain text, user always replies
        provider = FakeProvider(["thinking..." for _ in range(10)])
        registry = default_registry()
        monkeypatch.setattr("builtins.input", lambda _: "continue")
        agent_loop("loop forever", provider, registry, max_iterations=3)
        out = capsys.readouterr().out
        assert "maximum iterations" in out.lower()

    def test_read_file_tool_integration(self, monkeypatch, capsys, tmp_path):
        """Agent can use the read_file tool end-to-end."""
        target = tmp_path / "info.txt"
        target.write_text("secret data here")
        action = json.dumps({"tool": "read_file", "args": {"path": str(target)}})
        provider = FakeProvider([
            f"I'll read the file.\n\nACTION: {action}",
            "DONE: read the file",
        ])
        registry = default_registry()
        inputs = iter(["y", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent_loop("read that file", provider, registry, max_iterations=5)
        out = capsys.readouterr().out
        assert "secret data here" in out
