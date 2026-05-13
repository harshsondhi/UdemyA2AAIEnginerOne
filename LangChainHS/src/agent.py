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
    """A structured plan for resolving a support request."""

    summary: str = Field(description="1-2 sentence summary of the issue")
    steps: list[str] = Field(description="Concrete steps the user should take")
    needs_human: bool = Field(description="True if a human should review before action")


def main():
    # Initialize the language model
    llm = init_chat_model("openai:gpt-4.1-mini")
    checkpointer = InMemorySaver()

    # Create the agent with the tool and middleware
    agent = create_agent(
        model=llm,
        tools=[get_weather, calculate, search_knowledge_base, get_user_id, send_email],
        
        system_prompt="""
              You are a helpful support assistant. Use tools when needed.
            
              """,
        checkpointer=checkpointer,   
        response_format=SupportActionPlan,   
        # middleware=[
        #     SummarizationMiddleware(),
        #     PIIMiddleware(),
        #     HumanInTheLoopMiddleware()
        # ]
    )

    invoke_config = {"configurable": {"thread_id": "harsh-demo-thread"}}

    # Run the agent with a sample input
    response = agent.invoke(
        {"messages": [
            {"role": "user", "content": "I cant reset my password"}
        ]},
        context=Context(user_id="user-12345"),
        config=invoke_config,
    )
    
    print(
        '\n ----FINAL ANSWER ----'
    )
    print(response['structured_response'])
    print(response['messages'][-1].content)
    
    # print(
    #     '\n ----FINAL ANSWER-1` ----'
    # )
    # print(response['messages'][-1].content)
    
    # print(
    #     '\n ======= FINAL trace ====='
    # )
    # for m in response['messages']:
    #     print(f"{type(m)}:  {m.content}")
    
    
    # response = agent.invoke(
    #     {"messages": [
    #         {"role": "user", "content": "What city i asked wearther for in my previous call"}
    #     ]},
    #     config=invoke_config,
    # )
    # print(
    #     '\n ----FINAL ANSWER -2----'
    # )
    # print(response['messages'][-1].content)
    
    # print(
    #     '\n ----FINAL ANSWER ALL----'
    # )
    # print(response['messages'])

if __name__ == "__main__":
    main()