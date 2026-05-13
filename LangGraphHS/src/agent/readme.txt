# The LLM says: "I need to call the 'add' tool"
AIMessage(
    content="",
    tool_calls=[{
        "name": "add",
        "args": {"a": 3, "b": 4},
        "id": "call_123abc"  # <--- This is the ID we must match
    }]
)

# The LLM says: "I need to call the 'add' tool"
AIMessage(
    content="",
    tool_calls=[{
        "name": "add",
        "args": {"a": 3, "b": 4},
        "id": "call_123abc"  # <--- This is the ID we must match
    }]
)

from langchain_core.messages import ToolMessage

# The result is passed back to the model
ToolMessage(
    content="7",              # The answer from the tool
    tool_call_id="call_123abc" # Matches the AI's request ID
)
