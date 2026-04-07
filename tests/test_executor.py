from pen_tester_agent.executor import ask_tool_approval


class TestAskToolApproval:
    def test_approve(self, monkeypatch):
        # "0" = Approve
        monkeypatch.setattr("builtins.input", lambda _: "0")
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is True
        assert args == {"command": "ls"}

    def test_reject(self, monkeypatch):
        # "1" = Reject
        monkeypatch.setattr("builtins.input", lambda _: "1")
        args, execute = ask_tool_approval("bash", {"command": "rm -rf /"})
        assert execute is False
        assert args is None

    def test_edit_valid_json(self, monkeypatch):
        # "2" = Edit, then provide new JSON
        inputs = iter(["2", '{"command": "ls -la"}'])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is True
        assert args == {"command": "ls -la"}

    def test_edit_empty_cancels(self, monkeypatch):
        # "2" = Edit, then empty to cancel
        inputs = iter(["2", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is False
        assert args is None

    def test_edit_invalid_json_then_cancel(self, monkeypatch):
        # "2" = Edit, invalid JSON, then "1" = Reject
        inputs = iter(["2", "not json", "1"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is False
        assert args is None

    def test_invalid_input_rejects(self, monkeypatch):
        # Invalid menu choice is treated as rejection (safe default)
        monkeypatch.setattr("builtins.input", lambda _: "x")
        args, execute = ask_tool_approval("bash", {"command": "ls"})
        assert execute is False
        assert args is None
