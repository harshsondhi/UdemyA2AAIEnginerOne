import os
from typing import Literal
from deepagents import create_deep_agent
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
from deepagents.backends import FilesystemBackend

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def internet_search(
    query: str,
    max_results: int=5,
    topic: Literal["general", "news", "finance"]="general",
    include_raw_content: bool=False
):
    
    """ Perform an internet search for a given query and return the results."""
    print(query, topic, max_results)
    
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic
    )
    
    
model = init_chat_model("openai:gpt-4.1-mini")

research_instructions = """
    Search the internet for relevant information.

    Use this tool to gather current, factual, and source-backed information
    from the web. The tool is optimized for research, summarization, trend
    analysis, news discovery, and financial information retrieval.
    
    Use files as a scratchpad: write notes to /notes.md and drafts to /draft.md files \n
    When you revise text, prefer edit_file instead of rewriting the whole file. 

    Args:
        query:
            Natural language search query describing the information needed.

        max_results:
            Maximum number of search results to return.
            Higher values increase coverage but may introduce more noise.

        topic:
            Search category that guides retrieval behavior.
            Supported values:
            - "general" : broad web research and evergreen topics
            - "news"    : recent events, announcements, and developments
            - "finance" : markets, companies, stocks, and financial data

    Best Practices:
        - Start broad, then refine queries iteratively.
        - Prefer recent and authoritative sources.
        - Cross-check important claims across multiple results.
        - Use topic="news" for time-sensitive information.
        - Use topic="finance" for market or company-related research.

    Returns:
        A list of structured search results containing titles, snippets,
        source information, and relevant URLs.
    """
    
agent = create_deep_agent(
    model=model, 
    system_prompt=research_instructions, 
    tools=[internet_search],
    backend=FilesystemBackend(root_dir=".", virtual_mode=True)
)
    

results = agent.invoke(
    {
        "messages":[
            {"role": "user", "content": "Create a notes.md file with 10 bullet points on the topic of blackhole ?"}
        ]
    }
)

print(results['messages'][-1].content)
    
