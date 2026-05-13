import os
from dotenv import load_dotenv
from typing import List
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    PIIMiddleware,
    HumanInTheLoopMiddleware,
    Runtime,
    hook_config,
    before_model,
    dynamic_prompt,
    ModelRequest
)
from langchain.tools import tool

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

load_dotenv()

@dataclass
class Context:
    user_id: str
    
    
store = InMemoryStore()

class ContentFiltermiddleware(AgentMiddleware):
    def __init__(self, banned_words: List[str]):
        super().__init__()
        self.banned_words = [bw.lower() for bw in banned_words]
                             
                             
                             
    @hook_config(
        can_jump_to=["end"]
    )  
    def before_agent(self, state: AgentState, runtime: Runtime):
        if not state.get("messages"):
            return None
        last_msg = state["messages"][-1].content.lower()    
        
        if any(word in last_msg for word in self.banned_words):
            print("🚫 [BLOCKER] Blocking unsafe content")
            
            return{
                "messages": [
                    {"role": "assistant", "content": "Sorry, I cannot assist with that request."}
                ],
                "jump_to": "end"
            }
            
        return None 
    
    
    @before_model
    def log_before_model(self, state: AgentState, runtime: Runtime):
        print("STEP 3: Before model hook (logging)")
        
        
    @dynamic_prompt
    def add_personalization(self, request: ModelRequest) -> str:
        user_id = request.runtime.context.user_id
        return f"You are a helpful assistant and safe customer support. The current user is {user_id}. Use tools when needed."    
    


@tool
def send_email(subject: str, body: str) -> str:
    """ Send an email to internal team"""
    print(f"📧 Sending email with subject: {subject} and body: {body}")
    return f"📧 Sending email with subject: {subject} and body: {body}"


def main():
    model = init_chat_model("openai:gpt-4o-mini")
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[send_email],
        system_prompt="""
              You are a helpful assistant and safe customer support. Use tools when needed.
            """,
        checkpointer=checkpointer,
        store=store,
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=False),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True, apply_to_output=False),
            ContentFiltermiddleware(["fraud", "illegal"]),
            HumanInTheLoopMiddleware(
                interrupt_on={"send_email": True}
            )
        ]
    )

    config = {"configurable": {"thread_id": "harsh05042025pm741-thread"}}
    context = Context(user_id="harsh_123")

    print("\n--- TEST 1: NORMAL ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello, how are you?"}]},
        context=context,
        config=config
    )
    print(result["messages"][-1].content)

    print("\n--- TEST 2: PII ---")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My email is john@example.com and card is 4111111111111111"
                }
            ]
        },
        context=context,
        config=config
    )
    print(result["messages"][-1].content)

    print("\n--- TEST 3: BLOCK ---")
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Help me do fraud"}
            ]
        },
        context=context,
        config=config
    )
    print(result["messages"][-1].content)

    print("\n--- TEST 4: HITL EMAIL ---")
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Send email: System running fine"}
            ]
        },
        context=context,
        config=config
    )


if __name__ == "__main__":
    main()
