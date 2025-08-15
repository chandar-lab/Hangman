from langchain_core.tools import tool

# --- Tool Definition ---

@tool
def overwrite_memory(new_memory: str) -> str:
    """
    Overwrites the working memory with the provided content.

    Args:
        new_memory: The full working memory string to set.

    Returns:
        The provided new_memory string unchanged.
    """
    print("---TOOL: overwrite_memory---")
    print("Overwriting memory with provided content.")
    return new_memory
