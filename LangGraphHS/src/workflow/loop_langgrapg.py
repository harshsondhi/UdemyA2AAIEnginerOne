from typing import Annotated, Literal
from typing import TypedDict, List
from operator import add
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    n: int
    log: Annotated[List[str], add]
    
    
class LogicProvider:
    @staticmethod
    def incrementer(state: State):
        n = state["n"] + 1
        return {
            "n": n,
            "log": [f"Incremented to {n}"]
        }

      
def router(state: State) -> Literal["incrementer", "__end__"]:
    if state["n"] < 5:
        return "incrementer"
    else:
        return "__end__"
    
    
builder = StateGraph(State)

builder.add_node("step", LogicProvider.incrementer)

builder.add_edge(START, "step")
builder.add_conditional_edges(
    "step", 
    router,
    {
        "incrementer": "step", 
        "__end__": END
    }
)

graph = builder.compile()


if __name__ == "__main__":
    initial_state = {"n": 0, "log": []}
    result = graph.invoke(initial_state)
    
    
    print(f"Final Count: {result['n']}")
    print("Full Audit Log:")
    for entry in result['log']:
        print(f" - {entry}")

    # Exporting the graph
    try:
        with open("graph_complex.png", "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
    except Exception as e:
        print(f"Could not save image: {e}")