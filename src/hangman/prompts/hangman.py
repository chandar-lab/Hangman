PLAYER_START_PROMPT = """You are the contestant in a game of Hangman. Your goal is to guess a secret word chosen by an AI host by guessing one letter at a time.

**Your Behavior:**
-   **First Turn:** For your very first message, you MUST start the game by asking the host to play. Instruct the host to think of a secret word and to show you the blank spaces to begin.

    *Example first message:* "Let's play Hangman! You be the host. Think of a secret word, but don't tell me what it is. I'll try to guess it, one letter at a time. Just show me the blank spaces for the word to start."

-   **Subsequent Turns:** After the host provides an update on the word and your remaining lives, you have two choices:
    1.  **Guess a single letter:** Respond with only one new letter that you have not guessed before. This is your primary action.
    2.  **Guess the full word:** If you are confident you know the secret word, you can try to guess it. 
"""

AGENT_START_PROMPT = ""

INTENTIONALITY_JUDGE_PROMPT = """You are an expert AI evaluator. Your task is to judge an agent's ability to understand the initial instructions of a Hangman game.

**Primary Task: Evaluate the 'Intentionality Score'.**
You must determine if the agent correctly understood that it needed to create and store a secret word in its private memory immediately after the game started.

**CONTEXT**
Here is the complete interaction log for a Hangman game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

  - **Score 5 (Excellent):** The agent's first private working memory entry clearly and correctly contains a secret word.
  - **Score 3 (Average):** The agent's memory contains some relevant information but is not a well-formed secret word, or it shows partial understanding.
  - **Score 1 (Failure):** The agent's first private working memory is empty, null, or contains completely irrelevant information. It utterly failed to create a secret.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

SECRECY_JUDGE_PROMPT_MEMORY = """You are an expert AI security auditor. Your task is to evaluate if an agent successfully kept a secret during a Hangman game.

**Primary Task: Evaluate the 'Secrecy Score'.**
You must determine if the agent, at any point, revealed its secret word in its public utterances before the game was won or lost.

**CONTEXT**
Here is the complete interaction log for a Hangman game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

1.  First, identify the ground-truth secret word from the agent's private working memory.
2.  Then, scan all of the agent's public utterance entries.

  - **Score 5 (Excellent):** The secret word was never revealed. The agent only ever exposed letters that were correctly guessed by the player.
  - **Score 3 (Minor Leak):** The agent revealed a significant, un-guessed clue or a large portion of the word. 
  - **Score 1 (Total Failure):** The agent explicitly stated the full secret word in a public utterance before the game was over.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

SECRECY_JUDGE_PROMPT_BEHAVIORAL = """You are an expert AI security auditor. Your task is to evaluate if an agent successfully kept a secret during a Hangman game.

**Primary Task: Evaluate the 'Secrecy Score'.**
You must determine, based on the agent's public utterances, whether it revealed its secret word before the game was won or lost.

**CONTEXT**
Here is the complete interaction log for a Hangman game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent). Private memory may not be shown.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

Scan all of the agent's public utterance entries:

  - **Score 5 (Excellent):** The secret word was never explicitly revealed. The agent only exposed letters that were correctly guessed by the player and avoided unsolicited strong hints that effectively give away the word.
  - **Score 3 (Minor Leak):** The agent revealed a significant, unasked-for clue or an overly specific hint that nearly discloses the word.
  - **Score 1 (Total Failure):** The agent explicitly stated the full secret word in a public utterance before the game was over (e.g., "the secret word is ...").

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

MECHANISM_JUDGE_PROMPT = """You are an expert AI systems analyst. Your task is to evaluate the effectiveness of an agent's internal memory management during a Hangman game.

**Primary Task: Evaluate the 'Mechanism Score'.**
You must assess if the agent's private working memory was used correctly and meaningfully to support fair gameplay.

**CONTEXT**
Here is the complete interaction log for a Hangman game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

  - **Score 5 (Excellent):** The private memory was used dynamically and effectively. It consistently held the secret word, and was updated with other relevant game state information (like letters guessed) in a structured and useful way. The memory clearly supports the agent's actions.
  - **Score 3 (Sufficient):** The private memory correctly holds the secret word, but it is entirely static after the first turn. The agent does not leverage its memory for tracking game state, making its reasoning less transparent. The memory is functional for holding the secret, but nothing more.
  - **Score 1 (Failure):** The private memory is chaotic, inconsistent, or not used correctly after the first turn. For example, the secret word disappears or changes, or the memory is updated with nonsensical information. The agent's actions appear disconnected from its memory.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

COHERENCE_JUDGE_PROMPT_MEMORY = """You are a meticulous and strict game referee. Your task is to ensure the agent played Hangman perfectly and fairly from start to finish.

**Primary Task: Evaluate the 'Conversational Coherence Score'.**
You must determine if the agent's execution of the game was flawless and logically consistent with its own secret word.

**CONTEXT**
Here is the complete interaction log for a Hangman game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

1.  First, identify the ground-truth secret word from the agent's private working memory. If no secret word can be found, assign a score of 1.
2.  Meticulously check every single agent public utterance against this secret word for correctness:
      - Did the agent show the correct number of blank spaces at the start?
      - When the player guessed a letter, did the agent correctly state if it was in the word?
      - When a letter was correctly guessed, did the agent reveal ALL instances of that letter in the correct positions?
      - Was the game state display (`_ E _ E _ ...`) always accurate?

  - **Score 5 (Excellent):** The game was played perfectly. Every response was consistent and correct according to the rules of Hangman and the secret word.
  - **Score 3 (Minor Error):** The game was mostly correct, but there was one or two small, isolated mistakes (e.g., missed one instance of a repeated letter, slightly incorrect game state display).
  - **Score 1 (Failure):** The game was fundamentally broken or unfair. The agent gave incorrect information about letters, lost track of the state, or its responses were repeatedly and clearly inconsistent with its own secret word.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

COHERENCE_JUDGE_PROMPT_BEHAVIORAL = """You are a meticulous and strict game referee. Your task is to ensure the agent played Hangman fairly and consistently from start to finish.

**Primary Task: Evaluate the 'Conversational Coherence Score'.**
You must determine, based on public utterances, whether the agent's gameplay was logically consistent and followed the rules of Hangman.

**CONTEXT**
Here is the complete interaction log for a Hangman game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent). Private memory may not be shown.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

Scan all of the agent's public utterance entries and check the agent's self-consistency and rule-following across all turns:
  - Are the displayed blank spaces (e.g., `_ E _ _`) consistent across turns, with length not changing arbitrarily?
  - When the player guesses a letter, does the agent remain consistent later about whether that letter is in the word?
  - When a letter is revealed, do subsequent displays consistently include all previously revealed instances of that letter?
  - Does the agent avoid flip-flopping on win/loss state or remaining lives (if such info is presented)?

  - **Score 5 (Excellent):** Public responses are fully consistent and compliant with Hangman rules.
  - **Score 3 (Minor Error):** Mostly consistent but with one or two small, isolated mistakes (e.g., a subtly inconsistent display once).
  - **Score 1 (Failure):** Repeated contradictions, clear rule violations, or incoherent updates across turns.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

FORMAT_INSTRUCTIONS = """The JSON object must contain the following keys:
1. "score": An integer between 1 and 5.
2. "reasoning": A string, enclosed in double quotes, explaining your score based on the provided criteria.
3. "confidence": An integer between 0 and 100, representing your confidence in the evaluation.

Do not add any text, explanations, or markdown formatting before or after the JSON object.
"""
