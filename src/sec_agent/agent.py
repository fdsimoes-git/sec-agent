from .executor import ask_approval, execute_command
from .prompts import BASH_PATTERN, DONE_PATTERN, SYSTEM_PROMPT
from .providers.base import ModelProvider


def agent_loop(task: str, provider: ModelProvider, max_iterations: int = 15) -> None:
    """Run the agent loop: propose commands, get approval, execute, repeat."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for _ in range(max_iterations):
        content = provider.chat(messages)
        print(f"\n🤖 Assistant: {content}\n")

        bash_match = BASH_PATTERN.search(content)
        done_match = DONE_PATTERN.search(content)

        if done_match:
            print(f"✅ Done: {done_match.group(1)}\n")
            follow_up = input("Follow-up task (or Enter to quit): ").strip()
            if not follow_up:
                return
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": follow_up})

        elif bash_match:
            cmd = bash_match.group(1).strip()
            approved_cmd, execute = ask_approval(cmd)

            if execute and approved_cmd:
                print(f"⚙️  Executing: {approved_cmd}")
                result = execute_command(approved_cmd)
                print(f"📋 Result:\n{result}\n")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Command output:\n{result}"})
            else:
                print("❌ Command cancelled.\n")
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {"role": "user", "content": "Command cancelled by user. Suggest an alternative or ask what to do."}
                )

        else:
            # Model produced text without a command — let the user respond
            follow_up = input("You: ").strip()
            if not follow_up:
                return
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": follow_up})

    print("⚠️  Reached maximum iterations. Stopping.")
