import json

from .executor import ask_tool_approval
from .prompts import ACTION_PATTERN, build_system_prompt
from .providers.base import ModelProvider
from .tools.base import ToolRegistry


def agent_loop(task: str, provider: ModelProvider, registry: ToolRegistry, max_iterations: int = 15) -> None:
    """Run the agent loop: propose tool calls, get approval, execute, repeat."""
    system_prompt = build_system_prompt(registry)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    for _ in range(max_iterations):
        content = provider.chat(messages)
        print(f"\n🤖 Assistant: {content}\n")

        action_match = ACTION_PATTERN.search(content)

        if action_match:
            # Parse the tool call JSON
            try:
                action = json.loads(action_match.group(1))
            except json.JSONDecodeError:
                print("⚠️  Could not parse tool call JSON. Asking LLM to retry.")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Your ACTION was not valid JSON. Please try again with correct JSON format."})
                continue

            tool_name = action.get("tool", "")
            args = action.get("args", {})

            # Look up the tool
            tool = registry.get(tool_name)
            if tool is None:
                available = ", ".join(t.name for t in registry.list_tools())
                print(f"⚠️  Unknown tool: {tool_name}")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Unknown tool '{tool_name}'. Available tools: {available}"})
                continue

            # Ask for approval (some tools like 'done' skip this)
            if tool.requires_approval:
                approved_args, execute = ask_tool_approval(tool_name, args)
            else:
                approved_args, execute = args, True

            if execute and approved_args is not None:
                print(f"⚙️  Executing: {tool_name}")
                result = tool.execute(**approved_args)

                # Handle termination
                if result.terminates:
                    print(f"✅ Done: {result.output}\n")
                    follow_up = input("Follow-up task (or Enter to quit): ").strip()
                    if not follow_up:
                        return
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": follow_up})
                else:
                    status = "✅" if result.success else "❌"
                    print(f"{status} Result:\n{result.output}\n")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Tool '{tool_name}' output:\n{result.output}"})
            else:
                print("❌ Tool call cancelled.\n")
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {"role": "user", "content": f"Tool call '{tool_name}' cancelled by user. Suggest an alternative or ask what to do."}
                )

        else:
            # Model produced text without a tool call — let the user respond
            follow_up = input("You: ").strip()
            if not follow_up:
                return
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": follow_up})

    print("⚠️  Reached maximum iterations. Stopping.")
