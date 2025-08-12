from typing import List, Optional
from langchain_core.tools import tool

# --- Tool Definition ---

# current_memory arg is not truly optional but it is handled in the tool invokation in the agent's implementations

@tool
def update_memory(deletions: List[int], insertions: List[str], current_memory: Optional[str] = None) -> str:
    """
    Updates the working memory by deleting specified line numbers and appending new items.
    
    Args:
        deletions: A list of line numbers to delete from the memory. Numbers are 1-based.
        insertions: A list of new strings to add to the end of the memory.
    
    Returns:
        The newly formatted working memory string.
    """
    print(f"---TOOL: update_memory---")
    print(f"Deletions: {deletions}, Insertions: {insertions}")

    # Start with an empty list for the new memory
    new_memory_items = []
    
    # 1. Parse the current memory and keep only the items NOT marked for deletion
    if current_memory:
        existing_items = [line.split(". ", 1)[1] for line in current_memory.strip().split("\n") if ". " in line]
        for i, item in enumerate(existing_items):
            # Keep the item if its 1-based index is NOT in the deletions list
            if (i + 1) not in deletions:
                new_memory_items.append(item)

    # 2. Perform insertions by extending the list of kept items
    if insertions:
        new_memory_items.extend(insertions)
    
    # 3. Re-format the final list into a numbered list string
    new_memory = "\n".join(f"{i+1}. {item}" for i, item in enumerate(new_memory_items))
    print(f"New Memory:\n{new_memory}")
    return new_memory