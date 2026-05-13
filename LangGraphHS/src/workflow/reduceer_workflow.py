from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
import random
from operator import add

class State(TypedDict):
    topic: str
    log: Annotated[List[str], add]
    
    
def planner(state: State):
    return {
        "log": state["log"] + [f"Planning how to joke about {state['topic']}"] 
    }   
    
    
def researcher(state: State):
    return {
        "log": state["log"] + [f"Researching jokes about {state['topic']}"] 
    }
    
def writer(state: State):
    return {
        "log": state["log"] + [f"Writing a joke about {state['topic']}"] 
    }
    
def reviewer(state: State):
    return {
        "log": state["log"] + [f"Reviewing the joke about {state['topic']}"] 
    }
    
builder = StateGraph(State)

builder.add_node("planner", planner)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)

# Flow
builder.add_edge(START, "planner")

# Fan-out (parallel branches)
builder.add_edge("planner", "researcher")
builder.add_edge("planner", "writer")

# Fan-in (both go into reviewer)
builder.add_edge("researcher", "reviewer")
builder.add_edge("writer", "reviewer")

builder.add_edge("reviewer", END)

graph = builder.compile()

result = graph.invoke({"topic": "reducers", "log": []})

print(result) 