from langchain.tools import tool
from langchain.chat_models import init_chat_model


from dotenv import load_dotenv

load_dotenv()


model = init_chat_model(model="openai:gpt-4.1-mini")




@tool
def add(a: int, b: int) -> int:
    """Add two integers together.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divide two integers.

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        float: The quotient of a divided by b.

    Raises:
        ZeroDivisionError: If b is zero.
    """
    return a / b


tools = [add,multiply,divide]

tools_by_name = {t.name: t for t in tools}

model_with_tools = model.bind_tools(tools)

