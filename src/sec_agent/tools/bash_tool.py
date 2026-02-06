import subprocess

from .base import Tool, ToolResult


class BashTool(Tool):
    """Execute shell commands on the local system."""

    name = "bash"
    description = "Run a shell command and return its output. Use for system commands, security tools (nmap, curl, dig, etc.), and general CLI operations."
    parameters = {
        "command": {"type": "string", "description": "The shell command to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)", "default": 60},
    }
    requires_approval = True

    def execute(self, **kwargs) -> ToolResult:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", 60)

        if not command:
            return ToolResult(output="Error: no command provided", success=False)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr
            return ToolResult(output=output if output else "(no output)")
        except subprocess.TimeoutExpired:
            return ToolResult(output=f"Error: command timed out after {timeout}s", success=False)
        except Exception as e:
            return ToolResult(output=f"Error: {e}", success=False)
