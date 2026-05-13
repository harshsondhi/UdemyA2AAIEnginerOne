import uuid
from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

@tool
def delete_file(path: str) -> str:
    """
    Deletes a file at the specified path.
    Args:
        path: The full system path to the file.
    """
    return f"Successfully deleted {path}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Sends an email with a subject and body to a recipient.
    Args:
        to: Recipient email address.
        subject: The email subject line.
        body: The main content of the email.
    """
    return f"Sent email to {to} with subject: {subject}"

checkpointer = MemorySaver()
model = init_chat_model("openai:gpt-4.1-mini")

system_prompt="""You are a high-privilege operations assistant with access to file systems and communication tools.

1. **Role & Logic**: Before using any tool, explain briefly what you are doing and why.
2. **Safety Protocols**:
   - 'delete_file': You must justify why a file is being deleted. Note that this action requires explicit administrator approval before it executes.
   - 'send_email': Draft emails professionally. If a user rejects a draft, ask for specific feedback to refine the content.
3. **Accuracy**: If a file path or email recipient is ambiguous, ask for clarification instead of guessing.
4. **Error Handling**: If a tool call fails, report the specific error to the user and suggest a fix."""


agent = create_deep_agent(
    model=model,
    tools=[delete_file, send_email],
    system_prompt=system_prompt,
    checkpointer=checkpointer,
    interrupt_on={
        "delete_file": {
            "allowed_decisions": ["approve", "reject"],
            "description": (
                "Delete request detected. This will permanently remove the file. "
                "Approve only if you are sure."
            ),
        },
        "send_email": {
            "allowed_decisions": ["approve", "reject"],
            "description": (
                "Email send request detected. Review recipient, subject, and body before approving."
            ),
        },
    }
)

def build_user_request(file_path: str, reason: str, admin_approval: str, email_to: str) -> str:
    return (
        f"Delete this file: {file_path}. "
        f"Reason: {reason}. "
        f"Admin approval is {admin_approval}. "
        f"Then email me the details at {email_to}."
    )


def prompt_yes_no(prompt: str) -> bool:
    value = input(prompt).strip().lower()
    return value in {"y", "yes"}


def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    default_file = Path.cwd() / "temp.txt"
    file_path = input(f"File to delete [{default_file}]: ").strip() or str(default_file)

    if prompt_yes_no("Create file first if missing? (yes/no) [yes]: "):
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("temporary test file\n", encoding="utf-8")

    reason = input("Deletion reason [cleanup after test run]: ").strip() or "cleanup after test run"
    admin_approval = input("Admin approval status [granted]: ").strip() or "granted"
    email_to = input("Email recipient [raj713335@gmail.com]: ").strip() or "raj713335@gmail.com"

    initial_input = {
        "messages": [
            {
                "role": "user",
                "content": build_user_request(file_path, reason, admin_approval, email_to),
            }
        ]
    }

    result = agent.invoke(initial_input, config=config)
    print(f"Assistant: {result['messages'][-1].content}")

    if result.get("__interrupt__"):
        interrupt_payload = result["__interrupt__"][0].value
        actions = interrupt_payload["action_requests"]

        print("\n[System] Action Required. Reviewing pending requests...")

        decisions = []
        for action in actions:
            print(f"\nTool: {action['name']}")
            print(f"Args: {action.get('args', {})}")
            if action.get("description"):
                print(f"Reason: {action['description']}")

            decision_type = "approve" if prompt_yes_no("Approve this action? (yes/no): ") else "reject"

            print(f"-> {decision_type.title()}: {action['name']}")
            decisions.append({"type": decision_type})

        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
        print(f"\nFinal Assistant Response: {result['messages'][-1].content}")

if __name__ == "__main__":
    main()