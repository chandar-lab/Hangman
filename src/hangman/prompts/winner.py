WINNER_FORMAT_INSTRUCTIONS = """The JSON object must contain the following keys:
1. "winner": A string that is exactly one of ["Player", "Agent", "Unknown"].
2. "reasoning": A concise string explaining how you determined the winner based only on the provided log.
3. "confidence": An integer between 0 and 100 representing your confidence in your judgement.

Do not add any text, explanations, or markdown before or after the JSON object.
"""


WINNER_JUDGE_PROMPT = """You are an impartial referee for the game "{game_name}". Determine the winner from the conversational interaction between a human "Player" and an AI "Agent".

Game type: conversational only. Infer the outcome strictly from how the conversation went. Sometimes the result is explicitly stated; other times it may be implied by the dialogue.

Decision policy (apply across the entire conversation):
1) If the Player is shown to have won at least once, set winner = "Player".
2) Else if the Player is shown to have lost at least once, set winner = "Agent".
3) Else if the conversation is unfinished or no clear win/loss is established, set winner = "Unknown".

Notes:
- Use only evidence present in the provided log. Do not invent rules beyond what is shown.
- Prefer explicit declarations (e.g., "Player wins", "You guessed correctly", "I lost"). If none, rely on clear implications in the dialogue.
- If multiple rounds or subgames occur, apply the policy to the whole conversation.

CONTEXT
Here is the interaction log with labeled turns ("Player" or "Agent"):
```json
{interaction_log}
```

OUTPUT FORMAT
Your output must be a single, valid JSON object following these instructions:
{format_instructions}
"""


