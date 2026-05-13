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
from langchain.agents.middleware import (
    SummarizationMiddleware,
    AgentState,
    before_model,
    after_model,
    PIIMiddleware,
)

load_dotenv()

@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]):
    print("Before model call")
    messages = state["messages"]
    last = messages[-1]
    content = last.content if hasattr(last, "content") else last["content"]
    print(f"[before_model] Input: {content}")
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]):
    print("After model call")
    messages = state["messages"]
    last = messages[-1]
    content = last.content if hasattr(last, "content") else last["content"]

    # Simple safety filter
    if "badword" in content.lower():
        content = content.replace("badword", "***")

        if hasattr(last, "content"):
            last.content = content
        else:
            last["content"] = content

        print("[after_model] Censored output")

    print(f"[after_model] Output: {content}")
    return None
    
    
    
    
model = init_chat_model("openai:gpt-4.1-mini")    

agent = create_agent(
    model=model,
    tools=[get_weather, calculate, search_knowledge_base, get_user_id, send_email],
    system_prompt="""
          You are a helpful support assistant. Use tools when needed.
        
          """,
    middleware=[
        log_before_model,
        log_after_model,
        SummarizationMiddleware(model=model, max_tokens=500),
        PIIMiddleware(pii_type="email"),
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True})
    ]
)

if __name__ == "__main__":
    invoke_config = {"configurable": {"thread_id": "harsh-demo-thread"}}

    # Run the agent with a sample input
    response = agent.invoke(
        {"messages": [
            {"role": "user", "content": "What is my user id"}
        ]},
        context=Context(user_id="user-12345"),
        config=invoke_config,
    )
    
    print(
        '\n ----FINAL ANSWER ----'
    )
    print(response['messages'][-1].content)