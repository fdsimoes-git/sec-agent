"""Basic arithmetic tool."""

from .base import Tool, ToolResult

OPERATIONS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "modulo": lambda a, b: a % b,
    "power": lambda a, b: a ** b,
}


class MathTool(Tool):
    """Perform basic math operations on two numbers."""

    name = "math"
    description = (
        "Perform a mathematical operation on two numbers. "
        f"Supported operations: {', '.join(OPERATIONS)}."
    )
    parameters = {
        "operation": {
            "type": "string",
            "description": (
                "The operation to perform: "
                f"{', '.join(OPERATIONS)}"
            ),
        },
        "a": {"type": "number", "description": "The first operand"},
        "b": {"type": "number", "description": "The second operand"},
    }
    requires_approval = False

    def execute(self, **kwargs) -> ToolResult:
        operation = kwargs.get("operation", "")
        a = kwargs.get("a")
        b = kwargs.get("b")

        if a is None or b is None:
            return ToolResult(
                output="Error: both operands 'a' and 'b' are required",
                success=False,
            )

        try:
            a = float(a)
            b = float(b)
        except (TypeError, ValueError):
            return ToolResult(output="Error: operands must be numbers", success=False)

        op_func = OPERATIONS.get(operation)
        if op_func is None:
            return ToolResult(
                output=(
                    f"Error: unknown operation '{operation}'. "
                    f"Supported: {', '.join(OPERATIONS)}"
                ),
                success=False,
            )

        try:
            result = op_func(a, b)
        except ZeroDivisionError:
            return ToolResult(output="Error: division by zero", success=False)

        # Display ints cleanly (e.g. 6.0 -> 6)
        if result == int(result):
            result = int(result)

        return ToolResult(output=str(result))
