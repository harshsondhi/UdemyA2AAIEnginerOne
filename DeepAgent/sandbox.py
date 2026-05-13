import modal
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain_modal import ModalSandbox
from dotenv import load_dotenv

load_dotenv()


model = init_chat_model("openai:gpt-4.1-mini")

#app = modal.App.lookup("production-agent-environment", create_if_missing=True)
app = modal.App.lookup("production-agent-environment")
modal_sandbox = modal.Sandbox.create(
    app=app,
    image=modal.Image.debian_slim().pip_install("pytest", "requests", "pandas", "hypothesis"),
    timeout=600 # Longer timeout for complex builds
)


backend = ModalSandbox(sandbox=modal_sandbox)

agent = create_deep_agent(
    model=model,
    system_prompt=(
        "You are a Senior Lead Engineer. Use the 'write_todos' tool to plan first. "
        "Delegate testing to your 'tester' sub-agent. All files are in /workspace. "
        "If pytest fails because of a missing dependency, install it and rerun tests automatically."
    ),
    backend=backend,
    subagents=[
        {
            "name": "tester",
            "description": "Write and run exhaustive pytest suites.",
            "system_prompt": "You are a QA specialist. Your only job is to write and run exhaustive pytest suites.",
        }
    ],
)


results = agent.invoke({
    "messages": [
        {"role": "user", "content": "create a small python package and run pytest"}
    ]
})


print(results["messages"][-1].content)
# Cleanup
modal_sandbox.terminate()