import pytest

from sec_agent.tools.math_tool import MathTool


@pytest.fixture
def math():
    return MathTool()


class TestMathTool:
    def test_metadata(self, math):
        assert math.name == "math"
        assert math.requires_approval is False

    def test_add(self, math):
        r = math.execute(operation="add", a=2, b=3)
        assert r.success is True
        assert r.output == "5"

    def test_subtract(self, math):
        r = math.execute(operation="subtract", a=10, b=4)
        assert r.output == "6"

    def test_multiply(self, math):
        r = math.execute(operation="multiply", a=3, b=7)
        assert r.output == "21"

    def test_divide(self, math):
        r = math.execute(operation="divide", a=10, b=4)
        assert r.output == "2.5"

    def test_divide_even(self, math):
        r = math.execute(operation="divide", a=10, b=2)
        assert r.output == "5"

    def test_divide_by_zero(self, math):
        r = math.execute(operation="divide", a=5, b=0)
        assert r.success is False
        assert "division by zero" in r.output.lower()

    def test_modulo(self, math):
        r = math.execute(operation="modulo", a=10, b=3)
        assert r.output == "1"

    def test_power(self, math):
        r = math.execute(operation="power", a=2, b=8)
        assert r.output == "256"

    def test_float_operands(self, math):
        r = math.execute(operation="add", a=1.5, b=2.5)
        assert r.output == "4"

    def test_negative_numbers(self, math):
        r = math.execute(operation="add", a=-3, b=5)
        assert r.output == "2"

    def test_unknown_operation(self, math):
        r = math.execute(operation="sqrt", a=4, b=0)
        assert r.success is False
        assert "unknown operation" in r.output.lower()

    def test_missing_operands(self, math):
        r = math.execute(operation="add")
        assert r.success is False
        assert "required" in r.output.lower()

    def test_missing_one_operand(self, math):
        r = math.execute(operation="add", a=5)
        assert r.success is False

    def test_string_operands_that_are_numbers(self, math):
        r = math.execute(operation="add", a="3", b="4")
        assert r.success is True
        assert r.output == "7"

    def test_invalid_operands(self, math):
        r = math.execute(operation="add", a="foo", b=2)
        assert r.success is False
        assert "must be numbers" in r.output.lower()
