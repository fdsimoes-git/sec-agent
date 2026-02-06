# sec-agent

A security assistant agent powered by local LLMs via [Ollama](https://ollama.com).

`sec-agent` gives a locally-running language model the ability to propose and execute shell commands on your machine — with your approval on every step. It's designed for security tasks like reconnaissance, scanning, log analysis, and code review, but works for any command-line workflow.

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- A model pulled (default: `qwen2.5-coder:7b`):
  ```
  ollama pull qwen2.5-coder:7b
  ```
- [uv](https://docs.astral.sh/uv/) (for running/developing)

## Usage

```bash
# Interactive mode — prompts you for a task
uv run sec-agent

# Pass a task directly
uv run sec-agent "scan open ports on 192.168.1.1"

# Use a different model
uv run sec-agent --model llama3.1:8b "review nginx access.log for suspicious requests"

# Limit iterations
uv run sec-agent --max-iterations 5 "enumerate subdomains of example.com"
```

Every command the agent proposes is shown to you first. You can approve (`y`), reject (`n`), or edit (`e`) before anything runs.

## How it works

1. You describe a task in natural language.
2. The agent (running locally via Ollama) reasons about the task and proposes a shell command.
3. You review and approve/edit/reject the command.
4. The output is fed back to the agent for the next step.
5. Repeat until the task is complete or you stop.

## Disclaimer

This tool executes shell commands on your machine. Always review proposed commands before approving them. Use responsibly and only on systems you own or have explicit authorization to test. The authors are not responsible for any misuse or damage.
