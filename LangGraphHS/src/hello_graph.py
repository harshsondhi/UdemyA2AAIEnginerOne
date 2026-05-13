from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import AIMessage, HumanMessage

def hello_node(state: MessagesState) :
    return {"messages": [AIMessage(content="Hello, how can I help you?")]}


builder = StateGraph(MessagesState)

builder.add_node("hello", hello_node)


builder.add_edge(START, "hello")
builder.add_edge("hello", END)

graph = builder.compile()

initial_input = {"messages": [HumanMessage(content="Hi there!")]}
result = graph.invoke(initial_input)

print(result["messages"][-1].content)