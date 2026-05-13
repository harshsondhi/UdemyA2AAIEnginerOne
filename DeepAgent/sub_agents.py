import os
from typing import Literal
from deepagents import create_deep_agent
from tavily import TavilyClient
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()



tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def internet_search(
    query: str,
    max_results: int=3,
    topic: Literal["general", "news", "finance"]="general",
):
    """
    Performs a real-time web search using the Tavily API.

    Args:
        query: The search terms or question to look up.
        max_results: The maximum number of search results to return. Defaults to 5.
        topic: The category of search to perform. 
            - "general": Standard web search.
            - "news": Recent news articles.
            - "finance": Financial data and market information.

    Returns:
        list[dict]: A list of search result dictionaries containing URLs, 
            snippets, and content.
    """
    print(f"Performing internet search for query: '{query}' with topic: '{topic}' and max_results: {max_results}")
    
    results = tavily_client.search(
        query,
        max_results=max_results,
        topic=topic
    )
    
    print(f"Received {len(results)} results from Tavily API.")
    results = tavily_client.search(query, max_results=max_results, topic=topic)
    
    return results

def save_report_to_disk(content: str, filename: str="notes.md"):
    """
    Persists a generated report or string content to the local filesystem.

    This function utilizes the FilesystemBackend context manager to ensure 
    secure and organized file writing within a designated base directory.

    Args:
        content: The text data or report body to be saved.
        filename: The desired name of the file (e.g., 'analysis.md'). 
            Defaults to "notes.md".

    Note:
        Files are stored relative to the './agent_files' base path.
    """
    with open(filename, "w") as f:
        f.write(content)
    return f"Successfully saved to {filename}"


researcher_subagent = {
    "name": "researcher",
    "description": "Performs deep research and returns structured notes.",
    "system_prompt": (
        "You are a Research Specialist. Your goal is to gather raw data.\n"
        "1. Use internet_search to find facts.\n"
        "2. Output a structured summary: 'RAW_NOTES: [data] SOURCES: [urls]'.\n"
        "Be technical and detailed."
    ),
    "tools": [internet_search],
}

writer_subagent = {
    "name": "writer",
    "description": "Consumes research notes to produce a formatted Markdown file.",
    "system_prompt": (
        "You are a Technical Writer. You will receive RAW_NOTES and SOURCES.\n"
        "Your job is to format them into a beautiful Markdown report.\n"
        "1. Use 10 specific bullet points.\n"
        "2. Use the 'save_report_to_disk' tool to finalize the work."
    ),
    "tools": [save_report_to_disk],
}


model = init_chat_model("openai:gpt-4o-mini")

agent = create_deep_agent(
    model=model,
    tools=[internet_search, save_report_to_disk],
    subagents=[researcher_subagent, writer_subagent],
    system_prompt=(
        "You are the Lead Coordinator. Follow this sequence strictly:\n"
        "STEP 1: Call the 'researcher' subagent to gather info on the user's topic.\n"
        "STEP 2: Take the EXACT output from the researcher and pass it to the 'writer' subagent.\n"
        "STEP 3: Tell the writer to save the result using their tool.\n"
        "Do not answer the user until the file is saved."
    ),
    backend=FilesystemBackend(root_dir="./agent_state", virtual_mode=False)
)

inputs = {
    "messages": [
        {
            "role": "user",
            "content": """ 
            I need a report on Black Holes. Get the latest research, 
            then write it to notes.md with 10 bullet points.
            """
        }
    ]
}

print("---------  Starting Agent stream  -----------")
for update in agent.stream(inputs, stream_mode="updates"):
    # This will show the hand-offs between lead, researcher, and writer
    print("\n[Update Block]:", update)

print("\n--- Process Complete ---")