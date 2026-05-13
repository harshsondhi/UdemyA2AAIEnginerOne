import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

load_dotenv()


CONFIG = {
    "tutor": {
        "url": "http://127.0.0.1:8000/mcp",
        "transport": "streamable_http"
    }
}


async def main():
    client = MultiServerMCPClient(CONFIG)
    tools = await client.get_tools()

    model = init_chat_model("openai:gpt-4.1-mini")
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are TutorAI. Explain clearly and quiz the user."
    )

    response = await agent.ainvoke(
        {"messages": [("user", "Explain quantum computing and quiz me")]}
    )
    print(response['messages'][-1].content)


if __name__ == "__main__":
    print("\n🚀 Running TutorAI...\n")
    asyncio.run(main())
    #asyncio.run(async_stateful())
