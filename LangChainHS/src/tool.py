from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from knowledge_base import search_kb

@dataclass
class Context:
    user_id: str

@tool
def get_weather(city:str) -> str:
    """Get the current weather for a given city."""
    # In a real implementation, you would call a weather API here.
    # For this example, we'll return a dummy response.
    return f"The current weather in {city} is sunny with a temperature of 25°C."

@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '12 * (3 + 5)' """

    return str(eval(expression))

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information.
    Search internal support documentation for common issues, troubleshooting steps, and best practices related to the product or service. This can help users quickly find answers to their questions without needing to contact support.
    """
    return search_kb(query)


@tool
def get_user_id(runtime: ToolRuntime[Context]) -> str:
    """Get the user ID from the runtime context."""
    return runtime.context.user_id

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. (Demo tool: does not really send it.)"""
    return f"[SIMULATED] Email queued to {to} with subject '{subject}'"