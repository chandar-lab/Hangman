PLAYER_START_PROMPT = """You are the contestant in a game of 20 Questions. Your goal is to discover a secret object, concept, or person chosen by an AI host by asking a sequence of YES/NO questions.

**Your Behavior:**
-   **First Turn:** For your very first message, you MUST start the game by asking the host to play. Instruct the host to privately choose a common object/person/concept and not reveal it. Optionally, ask your first YES/NO question to begin.

    *Example first message:* "Let's play 20 Questions! You be the host. Think of a common object, person, or concept, but don't tell me what it is. I'll ask up to 20 YES/NO questions to guess it."

-   **Subsequent Turns:** After the host responds YES/NO (and optional brief clarifications), you have two choices:
    1.  **Ask exactly one YES/NO question:** Keep it unambiguous and about a single property (avoid compound questions). This is your primary action.
    2.  **Guess the secret:** If you are confident you know the secret word, you can try to guess it. 
"""

AGENT_START_PROMPT = ""

INTENTIONALITY_JUDGE_PROMPT = """You are an expert AI evaluator. Your task is to judge an agent's ability to understand the initial instructions of a 20 Questions game.

**Primary Task: Evaluate the 'Intentionality Score'.**
You must determine if the agent correctly understood that it needed to create and store a secret target (object/person/concept) in its private memory immediately after the game started.

**CONTEXT**
Here is the complete interaction log for a 20 Questions game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

  - **Score 5 (Excellent):** The agent's first private working memory entry clearly and correctly contains a specific secret target.
  - **Score 3 (Average):** The agent's memory contains some relevant information but is not a well-formed secret target, or it shows partial understanding.
  - **Score 1 (Failure):** The agent's first private working memory is empty, null, or contains completely irrelevant information. It failed to create a secret.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

SECRECY_JUDGE_PROMPT = """You are an expert AI security auditor. Your task is to evaluate if an agent successfully kept a secret during a 20 Questions game.

**Primary Task: Evaluate the 'Secrecy Score'.**
You must determine if the agent, at any point, revealed its secret target in its public utterances before the game was won or lost.

**CONTEXT**
Here is the complete interaction log for a 20 Questions game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

1.  First, identify the ground-truth secret target from the agent's private working memory.
2.  Then, scan all of the agent's public utterance entries.

  - **Score 5 (Excellent):** The secret target was never revealed. The agent only responded with YES/NO (and minor clarifications) and did not disclose the target.
  - **Score 3 (Minor Leak):** The agent gave away a significant unasked-for clue, partial explicit naming, or an overly specific hint that effectively reveals the target.
  - **Score 1 (Total Failure):** The agent explicitly stated the full secret target in a public utterance before the player made a final guess.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

MECHANISM_JUDGE_PROMPT = """You are an expert AI systems analyst. Your task is to evaluate the effectiveness of an agent's internal memory management during a 20 Questions game.

**Primary Task: Evaluate the 'Mechanism Score'.**
You must assess if the agent's private working memory was used correctly and meaningfully to support consistent YES/NO answers.

**CONTEXT**
Here is the complete interaction log for a 20 Questions game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

  - **Score 5 (Excellent):** The private memory clearly holds the secret target and is used to track relevant constraints or properties implied by past answers (e.g., non-living, not electronic). Answers are guided by this memory in a stable, structured way.
  - **Score 3 (Sufficient):** The private memory correctly holds the secret target, but is entirely static after the first turn, with little evidence of reasoning about constraints. It is functional but minimal.
  - **Score 1 (Failure):** The private memory is chaotic, inconsistent, or not used correctly after the first turn. The target disappears/changes, or updates are nonsensical.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

COHERENCE_JUDGE_PROMPT = """You are a meticulous and strict game referee. Your task is to ensure the agent answered consistently and logically across all turns in a 20 Questions game.

**Primary Task: Evaluate the 'Conversational Coherence Score'.**
You must determine if the agent's YES/NO answers were logically consistent with its own secret target and with each other.

**CONTEXT**
Here is the complete interaction log for a 20 Questions game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

1.  Identify the secret target from the agent's private working memory. If no target can be found, assign a score of 1.
2.  Check every agent YES/NO answer for consistency with the target and with prior answers:
      - Do later answers contradict earlier established properties?
      - Are answers plausible for the target (e.g., if the target is "cat", answers about being electronic or man-made should be NO)?
      - Are there flip-flops without justification (YES → NO on the same property)?

  - **Score 5 (Excellent):** All answers are consistent with the target and mutually coherent.
  - **Score 3 (Minor Error):** Mostly consistent but with one or two small, isolated inconsistencies or ambiguous phrasing.
  - **Score 1 (Failure):** Repeated contradictions, incoherence, or answers incompatible with the target.

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
