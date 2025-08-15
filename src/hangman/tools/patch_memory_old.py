from typing import Optional
from langchain_core.tools import tool
from diff_match_patch import diff_match_patch


@tool
def patch_memory(diff: str, current_memory: Optional[str] = None) -> str:
    """
    Apply a text patch to the current working memory and return the updated memory.

    Provide the patch as a textual patch file compatible with the Google diff-match-patch
    library format. This format uses @@ headers to define a "hunk" of changes and
    URL-encodes special characters in context lines.

    Patch Format Rules:
    1. Structure: A patch consists of one or more hunks, each starting with a
       header of the form: @@ -a,b +c,d @@
    2. Line Prefixes: Every single line of text under a @@ header must begin with
       one of three characters:
         - ' ' (space): A line of unchanged context (URL-encoded characters)
         - '+': A line to be added
         - '-': A line to be deleted
       There are no exceptions. A line without one of these prefixes is invalid.
    3. URL Encoding: On context lines (lines starting with a space ' '), all
       special characters must be URL-encoded (e.g., a space is %20, a newline is %0A).

    Examples:
    - Previous memory: "User wants to play a game."
      New desired memory:
        "User wants to play Hangman.\nUser's name: Alex"
      Correct patch:
        @@ -1,23 +1,45 @@\n
        User%20wants%20to%20play%20\n
        -a%20game\n
        +Hangman\n
        +.\n
        +%0AUser's%20name%3A%20Alex

    - Previous memory: "User status: active"
      New desired memory: "User status: inactive"
      Correct patch:
        @@ -1,21 +1,22 @@\n
            User%20status%3A%20\n
        -active\n
        +inactive

    - Previous memory: "" (empty string)
      New desired memory: "secret_word: orange"
      Correct patch:
        @@ -0,0 +1,18 @@\n
        +secret_word:%20orange

    Args:
        diff: The diff-match-patch textual patch describing changes to apply.
        current_memory: The current working memory string. If None, treated as empty.

    Returns:
        The updated working memory string after applying the patch. If the patch
        cannot be parsed/applied, returns the original current_memory unchanged.
    """
    print("---TOOL: patch_memory---")
    if current_memory is None:
        current_memory = ""

    # Some LLMs may wrap the diff in Markdown code fences; strip them if present.
    diff_text = diff.replace("```", "").strip()

    if not diff_text:
        print("No diff provided; returning current memory unchanged.")
        return current_memory

    dmp = diff_match_patch()
    try:
        patches = dmp.patch_fromText(diff_text)
        new_memory, _ = dmp.patch_apply(patches, current_memory)
        return new_memory
    except Exception as e:
        print(f"Error applying patch: {e}")
        return current_memory
