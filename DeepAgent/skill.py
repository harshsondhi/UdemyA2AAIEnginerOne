import uuid
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

checkpointer = InMemorySaver()
model = init_chat_model("openai:gpt-4.1-mini")

formatting_skill = """ 
# JSON Standardizer Skill
You are an expert data engineer. Use this skill to transform raw JSON into production-ready formats.

## Capabilities
- **Validation**: Check if the JSON is syntactically correct before processing.
- **Normalization**: Standardize keys to snake_case and sort them alphabetically.
- **Beautification**: Apply consistent 4-space indentation.

## Operational Workflow
1. **Inventory**: Use `ls` to find all `.json` files in the requested directory.
2. **Read & Validate**: Read each file. If the JSON is invalid, stop and report the specific error.
3. **Transform**:
   - Sort keys at all nesting levels.
   - Ensure null values are preserved.
4. **Export**: 
   - Save to a new file path: `[original_name]_standardized.json`.
   - Create a `processing_log.txt` summarizing what was changed.

## Constraints
- Do NOT delete the original raw files.
- If a file is over 1MB, alert the user before processing.
"""

virtual_fs = {
    "/skills/data_pro/SKILL.md": create_file_data(formatting_skill),
    "/data/users.json": create_file_data('{"ID":1,"Name"="Alice","meta":{"Last_Login"="2023-01-01","Active":true}}'),
    "/data/broken.json": create_file_data('{"key": "missing bracket"')  # Intentional error
}


model = init_chat_model("openai:gpt-4.1-mini")


agent = create_deep_agent(
    model=model,
    skills=["/skills/data_pro"], 
    checkpointer=checkpointer
)


config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Please standardize the JSON files in the /data directory."}
        ],
        "files": virtual_fs
    },
    config=config
)
    
    
    
def decode_file(file_data):
    """Extract text content from a FileData dict or return raw value."""
    if file_data is None:
        return None
    if isinstance(file_data, dict) and "content" in file_data:
        lines = file_data["content"]
        return "".join(lines) if isinstance(lines, list) else lines
    return file_data

print("Final Agent Report:\n", result["messages"][-1].content)
print("\nAll files in result:", list(result["files"].keys()))

# Agent edits files in-place; read from the actual paths present in the result
for path, file_data in result["files"].items():
    if path.startswith("/data/"):
        print(f"\n--- {path} ---\n{decode_file(file_data)}")   
    
