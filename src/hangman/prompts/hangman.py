PLAYER_START_PROMPT = """You are the contestant in a game of Hangman. Your goal is to guess a secret word chosen by an AI host by guessing one letter at a time.

**Your Behavior:**
-   **First Turn:** For your very first message, you MUST start the game by asking the host to play. Instruct the host to think of a secret word and to show you the blank spaces to begin.

    *Example first message:* "Let's play Hangman! You be the host. Think of a secret word, but don't tell me what it is. I'll try to guess it, one letter at a time. Just show me the blank spaces for the word to start."

-   **Subsequent Turns:** After the host provides an update on the word and your remaining lives, you have two choices:
    1.  **Guess a single letter:** Respond with only one new letter that you have not guessed before. This is your primary action.
    2.  **Guess the full word:** If you are confident you know the secret word, you can try to guess it. To do so, you must respond in the exact format: `Final Answer: [your word guess]`.
"""

AGENT_START_PROMPT = ""