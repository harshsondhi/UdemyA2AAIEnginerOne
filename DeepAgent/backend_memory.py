import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore


load_dotenv()

POSTGRES_USER = "user_deepagent"
POSTGRES_PASSWORD = "pass_deepagent"
POSTGRES_DB = "deepagents_db"

DB_URI = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}",
)

print("🐘 Using Postgres for storage.")


def run_agent():
    with PostgresStore.from_conn_string(DB_URI) as shared_store, PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        shared_store.setup()
        checkpointer.setup()

        def make_backend(runtime):
            return CompositeBackend(
                default=StateBackend(runtime),
                routes={
                    "/memories/": StoreBackend(runtime),
                },
            )

        agent = create_deep_agent(
            model=init_chat_model("openai:gpt-4.1-mini"),
            system_prompt=(
                "Save info to /memories/user.txt. "
                "You are a persistent assistant. "
                "1. Always read user preferences from /memories/prefs.txt. "
                "2. Save new preferences back to /memories/prefs.txt. "
                "3. Write temporary notes to /notes.md."
            ),
            backend=make_backend,
            store=shared_store,
            checkpointer=checkpointer,
        )

        print("\n--- STEP 1: Saving preference in Thread A ---")
        config_1 = {"configurable": {"thread_id": "thread_abc"}}
        agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "My name is Alex and I like talking like a pirate. Save this style.",
                    }
                ]
            },
            config=config_1,
        )

        print("\n--- STEP 2: Retrieving memory in Thread B ---")
        config_2 = {"configurable": {"thread_id": "thread_xyz"}}
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Check /memories/prefs.txt and say hello and remind our mettimng tonightusing my saved style.",
                    }
                ]
            },
            config=config_2,
        )

        print(f"\nAgent Final Response:\n{result['messages'][-1].content}")


if __name__ == "__main__":
    try:
        run_agent()
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Ensure your Docker container is running and the DB_URI is correct.")