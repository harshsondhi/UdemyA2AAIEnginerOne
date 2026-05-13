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

@dynamic_prompt
def personalize_by_role(request:ModelRequest) -> str:
    role = request.runtime.context.get("user_role", "guest")
    if role == "user":
        return "You are a helpful assistant that provides information to users. Use tools when needed."
    elif role == "admin":
        return "You are a system administrator assistant. Provide detailed technical information and use tools when needed."
    else:
        return "You are a helpful assistant. Use tools when needed."
    
    
model = init_chat_model("openai:gpt-4.1-mini")

agent = create_agent(
    model=model,
    tools=[get_weather, calculate, search_knowledge_base, get_user_id, send_email],
    middleware=[personalize_by_role]
)    


admin_output = agent.invoke(
    {"messages": [("user", "What is an IP address?")]},
    context={"user_role": "admin"},
    config={"configurable": {"thread_id": "dynamic-demo-thread"}}
)

print(f"Admin Response: {admin_output['messages'][-1].content}")