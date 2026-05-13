import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Callable, Any

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import AgentState, before_model, after_model, wrap_model_call, ModelRequest, \
    ModelResponse, dynamic_prompt, PIIMiddleware, AgentMiddleware, hook_config, HumanInTheLoopMiddleware

from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.types import Command

from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

from tool import get_weather, calculate, search_knowledge_base, get_user_id, send_email,Context

load_dotenv()

class SupportActionPlan(BaseModel):
    summary: str = Field(description="1-2 sentence summary of the issue")
    steps: list[str] = Field(description="Concrete steps the user should take")
    needs_human: bool = Field(description="True if a human should review before action")
    
    
@dataclass
class Context:
    user_id: str   
    
store = InMemoryStore()    
    
@tool
def save_issue(issue: str, runtime: ToolRuntime[Context]) -> str:  
    """
    Save the user's support issue into memory for later retrieval.

    Use this tool when the user describes a problem (e.g., login issues, billing issues).
    This ensures the issue can be referenced in future steps.
    """
    runtime.store.put(
        ("support",),
        runtime.context.user_id,
        {"issue": issue}
    )
    return "Issue saved successfully."


@tool
def get_issue(runtime: ToolRuntime[Context]) -> str:
    """
    Retrieve the user's previously saved support issue.

    Use this tool when you need to understand the user's problem before providing a solution.
    """
    data = runtime.store.get(("support",), runtime.context.user_id)
    return data.value["issue"] if data else "No issue found."


def main():
    model = init_chat_model("openai:gpt-4.1-mini")
    #checkpointer = InMemorySaver()
    
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_knowledge_base, get_user_id, send_email, save_issue, get_issue],
        store=store,
        #checkpointer=checkpointer,
       system_prompt="""
        You are a support assistant.

        Workflow:
        1. If the user describes a problem → call save_issue
        2. When solving → call get_issue first
        3. Then call kb_search for solutions
        4. Return a structured plan
        """,
        response_format=SupportActionPlan
    )
    
    config = {"configurable": {"thread_id": "harsh-05032026pm955-thread"}}
    # -------- TEST --------

    # -------- TEST --------

    # Step 1: Save issue
    agent.invoke(
        {"messages": [{"role": "user", "content": "I cannot reset my password"}]},
        context=Context(user_id="user-1")
    )

    # Step 2: Solve issue
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Help me fix it"}]},
        context=Context(user_id="user-1")
    )

    print("\n--- STRUCTURED OUTPUT ---")
    print(result["structured_response"])

    print("\n--- FINAL MESSAGE ---")
    print(result["messages"][-1].content)
    
    
if __name__ == "__main__":
    main()    