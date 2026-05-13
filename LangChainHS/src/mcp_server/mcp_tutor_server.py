from mcp.server.fastmcp import FastMCP
import httpx
import random
import urllib.parse

mcp = FastMCP("TutorAI", host="127.0.0.1", port=8000)

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_SEARCH = "https://en.wikipedia.org/w/rest.php/v1/search/title"

async def fetch_wiki_summary(topic: str) -> str:
    async with httpx.AsyncClient() as client:
        try:
            formatted = urllib.parse.quote(topic.replace(" ", "_"))
            url = WIKI_SUMMARY + formatted

            res = await client.get(url)

            if res.status_code == 200:
                data = res.json()
                return data.get("extract", "No summary available.")

            # 🔁 Fallback: search API
            search_res = await client.get(
                WIKI_SEARCH,
                params={"q": topic, "limit": 1}
            )

            if search_res.status_code == 200:
                results = search_res.json().get("pages", [])
                if results:
                    best_title = results[0]["title"]
                    retry_url = WIKI_SUMMARY + urllib.parse.quote(best_title)
                    retry_res = await client.get(retry_url)

                    if retry_res.status_code == 200:
                        return retry_res.json().get("extract", "No summary available.")

            return f"No good Wikipedia match found for '{topic}'."

        except Exception as e:
            return f"Error fetching topic: {str(e)}"
        
        
@mcp.tool()
async def explain_concept(topic: str) -> str:
    """
    Explain a topic using real-world knowledge from Wikipedia.

    Use this tool when:
    - The user asks for a definition or explanation
    - You need factual background information
    - You want reliable, neutral knowledge

    Args:
        topic (str): The concept or subject to explain.
                     Can be natural language (e.g., "machine learning", "quantum physics").

    Returns:
        str: A concise explanation of the topic.

    Example:
        explain_concept("machine learning")
    """
    return await fetch_wiki_summary(topic)



@mcp.tool()
def quiz_me(topic: str, difficulty: str = "easy") -> str:
    """
    Generate a quiz question about a topic.

    Use this tool when:
    - You want to test the user's understanding
    - You are in teaching mode
    - You need to continue an interactive session

    Args:
        topic (str): The subject to quiz the user on.
        difficulty (str): Difficulty level ("easy", "medium", "hard").

    Returns:
        str: A question related to the topic.

    Example:
        quiz_me("neural networks", "medium")
    """
    questions = [
        f"What is {topic}?",
        f"Why is {topic} important?",
        f"Give an example of {topic}.",
        f"Explain {topic} in simple terms.",
    ]
    return random.choice(questions)

@mcp.tool()
def give_hint(topic: str) -> str:
    """
    Provide a helpful hint to guide the user toward an answer.

    Use this tool when:
    - The user is struggling
    - You want to avoid giving the full answer
    - You are using a Socratic teaching approach

    Args:
        topic (str): The concept the user is working on.

    Returns:
        str: A guiding hint related to the topic.

    Example:
        give_hint("recursion")
    """
    return f"Think about the core idea and real-world use of {topic}."

@mcp.tool()
def evaluate_answer(question: str, answer: str) -> str:
    """
    Evaluate a student's answer and provide feedback.

    Use this tool when:
    - The user answers a question
    - You need to give feedback
    - You want to continue a tutoring loop

    Args:
        question (str): The original question asked.
        answer (str): The user's response.

    Returns:
        str: Feedback on the answer quality.

    Behavior:
        - Short answers → ask for more detail
        - Longer answers → positive reinforcement

    Example:
        evaluate_answer(
            question="What is machine learning?",
            answer="It is a way for machines to learn from data."
        )
    """
    if len(answer.strip()) < 15:
        return "Too short—expand your explanation."

    return "Good answer! You're on the right track."


if __name__ == "__main__":
    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    #mcp.run(transport="http")


