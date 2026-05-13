from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city (toy example)."""
    return f"It's always sunny in {city}"


model = init_chat_model("openai:gpt-4.1-mini")


agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistance."
)

result = agent.invoke({
    "messages":[
        {"role": "user", "content": "What's the weather in Paris?"}
    ]
})

print(result['messages'][-1].content)