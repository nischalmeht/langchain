"""Tools in LangChain: built-in tools, custom tools, and toolkits.

Imported from the Colab notebook tools-in-langchain.ipynb.

pip install langchain langchain-core langchain-community pydantic duckduckgo-search langchain_experimental
"""

from typing import Type

from langchain_community.tools import DuckDuckGoSearchRun, ShellTool
from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Built-in tool - DuckDuckGo Search
# ---------------------------------------------------------------------------
search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("top news in india today")
print(results)

print(search_tool.name)
print(search_tool.description)
print(search_tool.args)


# ---------------------------------------------------------------------------
# Built-in tool - Shell Tool
# ---------------------------------------------------------------------------
shell_tool = ShellTool()

results = shell_tool.invoke("ls")
print(results)


# ---------------------------------------------------------------------------
# Custom tools - Method 1: @tool decorator
# ---------------------------------------------------------------------------
# Step 1 - create a function, Step 2 - add type hints, Step 3 - add @tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


result = multiply.invoke({"a": 3, "b": 5})
print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())


# ---------------------------------------------------------------------------
# Custom tools - Method 2: StructuredTool
# ---------------------------------------------------------------------------
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")


def multiply_func(a: int, b: int) -> int:
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput,
)

result = multiply_tool.invoke({"a": 3, "b": 3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)


# ---------------------------------------------------------------------------
# Custom tools - Method 3: BaseTool class
# ---------------------------------------------------------------------------
# arg schema using pydantic
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a": 3, "b": 3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)

print(multiply_tool.args)


# ---------------------------------------------------------------------------
# Toolkit
# ---------------------------------------------------------------------------
# Custom tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


class MathToolkit:
    def get_tools(self):
        return [add, multiply]


toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "=>", tool.description)
