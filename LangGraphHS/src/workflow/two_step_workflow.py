from typing_extensions import TypedDict, List
import random

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    refined_topic: str
    tone: str
    drafts: List[str]
    scores: List[int]
    best_joke: str
    attempts: int
    retry: bool
    
    
def refine_topic(state: State):
    refined = f"{state['topic']} with a twist of absurd humor"
    return {
        "refined_topic": refined,
        "attempts": state.get("attempts", 0)
    }
    
def classify_tone(state: State):
    # Simple heuristic
    if "work" in state["topic"]:
        tone = "sarcastic"
    elif "science" in state["topic"]:
        tone = "nerdy"
    else:
        tone = "playful"

    return {"tone": tone}
   
   
def generate_drafts(state: State):
    topic = state["refined_topic"]
    tone = state["tone"]

    drafts = [
        f"[{tone}] Joke 1 about {topic}",
        f"[{tone}] Joke 2 about {topic}",
        f"[{tone}] Joke 3 about {topic}",
    ]
    return {"drafts": drafts}

def score_drafts(state: State):
    # Simulated scoring
    scores = [random.randint(1, 10) for _ in state["drafts"]]
    return {"scores": scores}

def pick_best(state: State):
    scores = state["scores"]
    drafts = state["drafts"]

    best_index = scores.index(max(scores))
    best_joke = drafts[best_index]

    return {"best_joke": best_joke}

def evaluate_quality(state: State):
    """Decide whether to retry or finish."""
    max_score = max(state["scores"])
    attempts = state["attempts"] + 1

    if max_score < 6 and attempts < 3:
        return {"attempts": attempts, "retry": True}
    else:
        return {"attempts": attempts, "retry": False}
    
    
def should_retry(state: State):
    return "retry" if state.get("retry", False) else "end"


builder = StateGraph(State)    

builder.add_node("refine_topic", refine_topic)
builder.add_node("classify_tone", classify_tone)
builder.add_node("generate_drafts", generate_drafts)
builder.add_node("score_drafts", score_drafts)
builder.add_node("pick_best", pick_best)
builder.add_node("evaluate_quality", evaluate_quality)

# Flow
builder.add_edge(START, "refine_topic")
builder.add_edge("refine_topic", "classify_tone")
builder.add_edge("classify_tone", "generate_drafts")
builder.add_edge("generate_drafts", "score_drafts")
builder.add_edge("score_drafts", "pick_best")
builder.add_edge("pick_best", "evaluate_quality")

builder.add_conditional_edges(
    "evaluate_quality",
    should_retry,
    {
        "retry": "generate_drafts",  # loop back
        "end": END
    }
)

graph = builder.compile()

result = graph.invoke({
    "topic": "quantum physics",
    "refined_topic": "",
    "tone": "",
    "drafts": [],
    "scores": [],
    "best_joke": "",
    "attempts": 0,
    "retry": False
})

print(result)