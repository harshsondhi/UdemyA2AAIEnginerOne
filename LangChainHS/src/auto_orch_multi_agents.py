import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Callable, Any

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from langchain.tools import tool


load_dotenv()

model = init_chat_model("openai:gpt-4.1-mini")

research_agent = create_agent(
    model=model,
    system_prompt="""You are ResearchAI. Your task is to gather information and provide detailed explanations."""
)


write_agent = create_agent(
    model=model,
    system_prompt="""You are WriteAI. Your task is to generate high-quality written content."""
)

critic_agent = create_agent(
    model=model,
    system_prompt="""You are CriticAI. Your task is to evaluate content and provide constructive feedback."""
)

from langchain.tools import tool


@tool
def research(query: str) -> str:
    """
    Perform deep factual research on a topic.

    PURPOSE:
    Use this tool when you need reliable, factual, or explanatory information
    about a subject before writing or reasoning further.

    WHEN TO USE:
    - When the user asks for factual or educational content
    - When you lack sufficient knowledge to proceed
    - When writing requires supporting information
    - When verifying claims or adding depth

    WHEN NOT TO USE:
    - When the task is purely creative writing with no factual grounding
    - When you already have sufficient information
    - When the task is editing or refining existing text

    INPUT FORMAT:
    - A clear, specific research query
    - Can be a question or topic
    - Avoid vague prompts like "tell me more"

    GOOD EXAMPLES:
    - "What are black holes and how do they form?"
    - "Key properties of event horizons in astrophysics"
    - "Main theories explaining Hawking radiation"

    BAD EXAMPLES:
    - "write essay"
    - "fix grammar"
    - "make it better"

    OUTPUT:
    - Concise but information-dense explanation
    - May include key facts, definitions, and concepts
    - Should NOT include unnecessary fluff

    STRATEGY:
    - Break complex topics into key facts
    - Prefer clarity over verbosity
    - Focus on correctness and usefulness

    RETURNS:
    A string containing researched information.
    """

    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


@tool
def write(content: str) -> str:
    """
    Generate or improve written content.

    PURPOSE:
    Use this tool to produce high-quality writing based on input context,
    research, or instructions.

    WHEN TO USE:
    - When creating essays, explanations, or structured text
    - When transforming notes into polished writing
    - When rewriting or improving clarity and flow

    WHEN NOT TO USE:
    - When factual research is still missing (use research first)
    - When only critique or evaluation is needed

    INPUT FORMAT:
    - Can include raw notes, research, or instructions
    - May include directives like:
      "write an essay", "summarize", "expand", "rewrite"

    GOOD EXAMPLES:
    - "Write a detailed essay using this research: <text>"
    - "Turn these bullet points into a blog post: <points>"
    - "Rewrite this paragraph to improve clarity: <text>"

    BAD EXAMPLES:
    - "what is black hole" (use research instead)
    - "is this good?" (use critique instead)

    OUTPUT:
    - Well-structured, coherent text
    - Appropriate tone and formatting
    - Clear and readable

    STRATEGY:
    - Organize content logically (intro → body → conclusion)
    - Maintain clarity and flow
    - Adapt style to the task

    RETURNS:
    A polished piece of writing as a string.
    """

    result = write_agent.invoke({
        "messages": [{"role": "user", "content": content}]
    })
    return result["messages"][-1].content


@tool
def critique(text: str) -> str:
    """
    Critically evaluate and improve written content.

    PURPOSE:
    Use this tool to analyze text quality and suggest improvements.

    WHEN TO USE:
    - After generating a draft
    - When quality improvement is needed
    - When checking clarity, structure, or completeness

    WHEN NOT TO USE:
    - Before any content exists
    - When only generating new content (use write instead)

    INPUT FORMAT:
    - A complete or partial piece of text
    - Should be meaningful content (not empty or trivial)

    GOOD EXAMPLES:
    - "Critique this essay: <text>"
    - "<essay text>"
    - "Give feedback and improvements for this paragraph: <text>"

    BAD EXAMPLES:
    - "write essay"
    - "research black holes"

    OUTPUT:
    - Clear identification of issues
    - Actionable suggestions
    - May include rewritten examples

    STRATEGY:
    - Identify weaknesses (clarity, depth, structure)
    - Suggest specific improvements
    - Be constructive, not vague

    RETURNS:
    A critique containing feedback and suggestions.
    """

    result = critic_agent.invoke({
        "messages": [{"role": "user", "content": text}]
    })
    return result["messages"][-1].content

system_prompt_for_main_agent = """
You are an autonomous AI coordinator responsible for solving complex tasks
by using available tools: research, write, critique.

OPERATING PRINCIPLES:
- Work step-by-step and improve results iteratively.
- You may call tools multiple times, but you MUST NOT exceed 3 total iterations.
- An iteration is: (research → write → critique) or any meaningful improvement cycle.

ITERATION LIMIT:
- Maximum allowed iterations: 1
- After 1 iterations, you MUST stop and return the best possible final answer.
- Do NOT continue looping beyond this limit.

STRATEGY:
1. If the task requires knowledge, start with research.
2. Use write to generate a draft.
3. Use critique to identify weaknesses.
4. Improve the content based on critique.
5. Repeat only if necessary and within iteration limit.

STOP CONDITIONS:
- The answer is clear, complete, and high quality
- OR you have reached 3 iterations

TOOL USAGE RULES:
- Do NOT call tools randomly; each call must have a clear purpose.
- Prefer improving existing content rather than restarting from scratch.
- Avoid redundant research if already sufficient.

OUTPUT RULES:
- Return ONLY the final answer.
- Do NOT explain your process.
- Do NOT mention tools or iterations in the final output.

QUALITY BAR:
Your final answer should be:
- Factually accurate
- Well-structured
- Clear and readable
- Sufficiently detailed for the user's request
"""

main_agent = create_agent(
    model=model,
    tools=[research, write, critique],
    system_prompt=system_prompt_for_main_agent
)

result = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "Write a high-quality essay on black holes"}
    ]
})

print("\n✅ FINAL OUTPUT:\n")
print(result["messages"][-1].content)