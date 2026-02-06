import json

from .context import ContextManager
from .executor import ask_tool_approval
from .prompts import ACTION_PATTERN, build_system_prompt
from .providers.base import ModelProvider
from .tools.base import ToolRegistry


def agent_loop(
    task: str,
    provider: ModelProvider,
    registry: ToolRegistry,
    max_iterations: int = 15,
    max_context_tokens: int = 6000,
) -> None:
    """Run the agent loop: propose tool calls, get approval, execute, repeat."""
    system_prompt = build_system_prompt(registry)
    ctx = ContextManager(system_prompt, task, max_context_tokens=max_context_tokens)

    for _ in range(max_iterations):
        content = provider.chat(ctx.get_messages())
        print(f"\n🤖 Assistant: {content}\n")

        action_match = ACTION_PATTERN.search(content)

        if action_match:
            # Parse the tool call JSON
            try:
                action = json.loads(action_match.group(1))
            except json.JSONDecodeError:
                print("⚠️  Could not parse tool call JSON. Asking LLM to retry.")
                ctx.add_assistant(content)
                ctx.add_user("Your ACTION was not valid JSON. Please try again with correct JSON format.")
                continue

            tool_name = action.get("tool", "")
            args = action.get("args", {})

            # Look up the tool
            tool = registry.get(tool_name)
            if tool is None:
                available = ", ".join(t.name for t in registry.list_tools())
                print(f"⚠️  Unknown tool: {tool_name}")
                ctx.add_assistant(content)
                ctx.add_user(f"Unknown tool '{tool_name}'. Available tools: {available}")
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
                    ctx.add_assistant(content)
                    ctx.add_user(follow_up)
                else:
                    status = "✅" if result.success else "❌"
                    print(f"{status} Result:\n{result.output}\n")
                    ctx.add_assistant(content)
                    ctx.add_tool_result(tool_name, result.output)
            else:
                print("❌ Tool call cancelled.\n")
                ctx.add_assistant(content)
                ctx.add_user(f"Tool call '{tool_name}' cancelled by user. Suggest an alternative or ask what to do.")

        else:
            # Model produced text without a tool call — let the user respond
            follow_up = input("You: ").strip()
            if not follow_up:
                return
            ctx.add_assistant(content)
            ctx.add_user(follow_up)

    print("⚠️  Reached maximum iterations. Stopping.")
