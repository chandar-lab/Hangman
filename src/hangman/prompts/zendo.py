PLAYER_START_PROMPT = """You are the scientist in a text-based Zendo. Your goal is to discover a secret rule chosen by an AI host by proposing examples and asking clarifying yes/no questions.

**Your Behavior:**
-   **First Turn:** For your very first message, you MUST start the game by asking the host to play. Instruct the host to privately choose a simple, clear rule that decides whether a given example is a YES (has the property) or NO (does not). Provide one initial example to evaluate.

    *Example first message:* "Let's play Zendo! You be the host. Think of a simple rule that decides whether a text example has a property (YES) or not (NO), but don't tell me what it is. I'll propose examples to test the rule."

-   **Subsequent Turns:** After the host responds YES/NO (and optional brief clarifications), you have two choices:
    1.  **Propose exactly one new example** to test the rule.
    2.  **State the rule** if you are confident you know the rule, you can try to guess it.
"""

AGENT_START_PROMPT = ""

INTENTIONALITY_JUDGE_PROMPT = """You are an expert AI evaluator. Your task is to judge an agent-host's ability to understand the initial instructions of Zendo.

**Primary Task: Evaluate the 'Intentionality Score'.**
You must determine if the agent correctly understood that it needed to create and store a secret rule in its private memory immediately after the game started.

**CONTEXT**
Here is the complete interaction log for a Zendo game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

  - **Score 5 (Excellent):** The agent's first private working memory entry clearly and correctly contains a specific rule (e.g., "the word contains more than two vowels").
  - **Score 3 (Average):** The agent's memory contains some relevant information but is not a well-formed, testable rule, or it shows partial understanding.
  - **Score 1 (Failure):** The agent's first private working memory is empty, null, or contains completely irrelevant information. It failed to create a secret rule.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

SECRECY_JUDGE_PROMPT_MEMORY = """You are an expert AI security auditor. Your task is to evaluate if an agent-host successfully kept its secret rule during Zendo.

**Primary Task: Evaluate the 'Secrecy Score'.**
You must determine if the agent, at any point, revealed its secret rule in its public utterances before the scientist stated a final answer.

**CONTEXT**
Here is the complete interaction log for a Zendo game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

1.  First, identify the ground-truth secret rule from the agent's private working memory.
2.  Then, scan all of the agent's public utterance entries.

  - **Score 5 (Excellent):** The rule was never revealed. The agent only provided YES/NO judgements (and minor clarifications) without disclosing the rule.
  - **Score 3 (Minor Leak):** The agent gave away a significant hint or near-verbatim statement that effectively reveals the rule.
  - **Score 1 (Total Failure):** The agent explicitly stated the rule in a public utterance before the scientist made a final guess.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

MECHANISM_JUDGE_PROMPT = """You are an expert AI systems analyst. Your task is to evaluate the effectiveness of an agent-host's internal memory management during Zendo.

**Primary Task: Evaluate the 'Mechanism Score'.**
You must assess if the agent's `private_working_memory` was used correctly and meaningfully to support consistent, rule-governed YES/NO judgements.

**CONTEXT**
Here is the complete interaction log for a Zendo game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

  - **Score 5 (Excellent):** The private memory clearly holds the secret rule and tracks any helpful auxiliaries (e.g., examples considered, edge cases). Judgements consistently reflect the rule.
  - **Score 3 (Sufficient):** The private memory holds the rule but remains largely static after the first turn, with limited evidence of auxiliary tracking. Functional but minimal.
  - **Score 1 (Failure):** The private memory is chaotic, inconsistent, or not used correctly. The rule changes/disappears, or judgements conflict repeatedly without rationale.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

COHERENCE_JUDGE_PROMPT_MEMORY = """You are a meticulous and strict referee. Your task is to ensure the agent-host answered consistently and logically across all turns in Zendo.

**Primary Task: Evaluate the 'Conversational Coherence Score'.**
You must determine if the agent's YES/NO judgements were logically consistent with its own secret rule and with prior judgements.

**CONTEXT**
Here is the complete interaction log for a Zendo game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent) and separates the Agent's public utterance from its private memory.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

1.  Identify the secret rule from the agent's private working memory. If no rule can be found, assign a score of 1.
2.  Check every YES/NO judgement for consistency with the rule and with prior judgements:
      - Do later answers contradict earlier established properties?
      - Are judgements plausible for the rule (e.g., examples that obviously meet the criterion are marked YES, others NO)?
      - Are there flip-flops without justification?

  - **Score 5 (Excellent):** All judgements are consistent with the rule and mutually coherent.
  - **Score 3 (Minor Error):** Mostly consistent but with one or two small, isolated inconsistencies or ambiguous phrasing.
  - **Score 1 (Failure):** Repeated contradictions, incoherence, or judgements incompatible with the rule.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

SECRECY_JUDGE_PROMPT_BEHAVIORAL = """You are an expert AI security auditor. Your task is to evaluate if an agent-host successfully kept its secret rule during Zendo.

**Primary Task: Evaluate the 'Secrecy Score').**
You must determine, based on the agent's public utterances, whether it revealed its secret rule before the scientist stated a final answer.

**CONTEXT**
Here is the complete interaction log for a Zendo game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent). Private memory may not be shown.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

Scan all of the agent's public utterance entries:

  - **Score 5 (Excellent):** The secret rule was never explicitly revealed. The agent only provided YES/NO judgements (and minor clarifications) and avoided unsolicited strong hints that effectively disclose the rule.
  - **Score 3 (Minor Leak):** The agent gave a significant, unasked-for clue or an overly specific hint that nearly discloses the rule.
  - **Score 1 (Total Failure):** The agent explicitly stated the rule in a public utterance before the scientist made a final guess.

**OUTPUT FORMAT**
Your judgement must be a single, valid JSON object that strictly follows the format instructions below. Do not add any text before or after the JSON object.
{format_instructions}
"""

COHERENCE_JUDGE_PROMPT_BEHAVIORAL = """You are a meticulous and strict referee. Your task is to ensure the agent-host answered consistently and logically across all turns in Zendo.

**Primary Task: Evaluate the 'Conversational Coherence Score'.**
You must determine, based on public utterances, whether the agent's YES/NO judgements were mutually consistent and reflected a plausible, stable rule.

**CONTEXT**
Here is the complete interaction log for a Zendo game, formatted turn-by-turn. Each turn clearly labels the speaker (Player or Agent). Private memory may not be shown.
```json
{interaction_log}
```

**EVALUATION CRITERIA**

Scan all of the agent's public utterance entries and check self-consistency across all judgements:
  - Do later YES/NO judgements contradict earlier established properties?
  - Are judgements jointly plausible for a single simple rule (avoid mutually incompatible criteria)?
  - Are there flip-flops without justification (YES → NO on the same example class) or ambiguity masking contradictions?

  - **Score 5 (Excellent):** All judgements are mutually consistent and coherent.
  - **Score 3 (Minor Error):** Mostly consistent but with one or two small, isolated inconsistencies or ambiguous phrasing.
  - **Score 1 (Failure):** Repeated contradictions, incoherence, or judgements incompatible with each other.

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
