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

from tool import get_weather, calculate, search_knowledge_base, get_user_id, send_email,Context

load_dotenv()

class SupportActionPlan(BaseModel):
    summary: str = Field(description="1-2 sentence summary of the issue")
    steps: list[str] = Field(description="Concrete steps the user should take")
    needs_human: bool = Field(description="True if a human should review before action")

def main():
    model = init_chat_model("openai:gpt-4.1-mini")
    checkpointer = InMemorySaver()
    
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_knowledge_base, get_user_id, send_email],
        checkpointer=checkpointer,
        system_prompt=""" 
        You are a support assistant.
        Always use kb_search tool before answering.
        Return structured plan.
        """,
        response_format=SupportActionPlan
    )
    
    config = {"configurable": {"thread_id": "harsh-05032026pm955-thread"}}
    # -------- TEST --------

    # Step 1 (user problem)
    agent.invoke(
        {"messages": [{"role": "user", "content": "I can't reset my password"}]},
        config=config
    )

    # Step 2 (follow-up)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "help me fix it"}]},
        config=config
    )

    print("\n--- STRUCTURED OUTPUT ---")
    print(result["structured_response"])

    print("\n--- FINAL MESSAGE ---")
    print(result["messages"][-1].content)
    
if __name__ == "__main__":
    main()    