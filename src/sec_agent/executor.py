import subprocess


def ask_approval(command: str) -> tuple[str | None, bool]:
    """Request user approval before executing a command."""
    print(f"\n⚠️  Proposed command: {command}")
    while True:
        response = input("Execute? (y/n/e to edit): ").lower().strip()
        if response == "y":
            return command, True
        elif response == "n":
            return None, False
        elif response == "e":
            edited = input("Enter edited command: ").strip()
            if edited:
                return edited, True
        else:
            print("Please type 'y', 'n', or 'e'")


def execute_command(command: str, timeout: int = 60) -> str:
    """Execute a shell command and return combined stdout/stderr."""
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
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"
