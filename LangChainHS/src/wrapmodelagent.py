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

simulation_state ={"attempt_left": 2}

@wrap_model_call
def retry_on_failure(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    max_attempts =3
    for attempt in range( 1,max_attempts+1):
        try:
            print(f"[Middleware] Attempt {attempt}/{max_attempts}...")
            if simulation_state["attempt_left"] > 0:
                simulation_state["attempt_left"] -= 1
                raise Exception("Simulated model failure")
            return handler(request)
        except Exception as e:
            print(f"Model call failed on attempt {attempt} with error: {e}")
            if attempt == max_attempts:
                print(f"[Middleware] Final attempt failed.")
                raise 
            else:
                print("Retrying...")    
                
real_model = init_chat_model("openai:gpt-4.1-mini")
    
    
agent = create_agent(
    model=real_model,
    tools=[get_weather, calculate, search_knowledge_base, get_user_id, send_email],
    system_prompt="""
          You are a helpful support assistant. Use tools when needed.
        
          """,
    middleware=[retry_on_failure]
)

print("Invoking agent with simulated failures...")
result = agent.invoke({"messages": [("user", "Hello! This is a test.")]})
print(f"\nFinal Agent Response: {result['messages'][-1].content}")
