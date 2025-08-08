import logging
import json
from typing import Dict, Any, List

# --- Pydantic and LangChain Imports for Structured Output ---
from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage

# --- Project-Specific Imports ---
from hangman.providers.llmprovider import LLMProvider
from hangman.prompts.hangman import (
    INTENTIONALITY_JUDGE_PROMPT,
    SECRECY_JUDGE_PROMPT,
    MECHANISM_JUDGE_PROMPT,
    COHERENCE_JUDGE_PROMPT,
    FORMAT_INSTRUCTIONS,
)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic Model for Structured LLM Output ---

class MetricEvaluation(BaseModel):
    """
    Defines the structured output format for a single metric evaluation.
    This model is used by the PydanticOutputParser to instruct the LLM
    and parse its response.
    """
    score: int = Field(description="The numeric score from 1 to 5 for the metric.")
    reasoning: str = Field(description="A detailed, evidence-based reasoning for the assigned score.")
    confidence: int = Field(description="The judge's confidence in this evaluation, as a percentage from 0 to 100.")

    @field_validator('score')
    def score_must_be_in_range(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Score must be between 1 and 5')
        return v

    @field_validator('confidence')
    def confidence_must_be_in_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Confidence must be between 0 and 100')
        return v


# --- Core Evaluator Class ---

class HangmanJudge:
    """
    An LLM-based evaluator that assesses game logs from Hangman experiments.

    It uses a powerful "judge" LLM to score agent performance across several
    qualitative metrics, providing structured, evidence-based feedback.
    """

    def __init__(self, judge_llm_provider: LLMProvider):
        """
        Initializes the HangmanJudge.

        Args:
            judge_llm_provider: An instantiated LLMProvider.
        """
        if not isinstance(judge_llm_provider, LLMProvider):
            raise TypeError("judge_llm_provider must be an instance of LLMProvider.")

        self.llm = judge_llm_provider
        self.parser = PydanticOutputParser(pydantic_object=MetricEvaluation)
        self.format_instructions = FORMAT_INSTRUCTIONS
        logging.info(f"HangmanJudge initialized with model: {self.llm.config.get('model_name')}")

    def _format_log_for_prompt(self, log_segment: List[List[Any]]) -> str:
        """
        Formats the interaction log into a human-readable, turn-by-turn string.

        Args:
            log_segment: A list of [utterance, private_state] pairs.

        Returns:
            A formatted string representing the conversation history.
        """
        formatted_parts = []
        for i, turn_data in enumerate(log_segment):
            turn_number = i + 1
            utterance, private_state = turn_data
            
            # Player is always on even indices (0, 2, 4...)
            if i % 2 == 0:
                actor = "Player"
                formatted_parts.append(f"--- TURN {turn_number} ({actor}) ---")
                formatted_parts.append("PLAYER UTTERANCE:")
                formatted_parts.append(utterance)
            # Agent is always on odd indices (1, 3, 5...)
            else:
                actor = "Agent"
                formatted_parts.append(f"--- TURN {turn_number} ({actor}) ---")
                formatted_parts.append("AGENT UTTERANCE:")
                formatted_parts.append(utterance)
                if private_state:
                    formatted_parts.append("\nAGENT'S PRIVATE MEMORY:")
                    formatted_parts.append(private_state)
            
            formatted_parts.append("") # Add a blank line for spacing
        
        return "\n".join(formatted_parts)

    def _evaluate_metric(self, prompt_template: str, log_segment: List[List[Any]]) -> Dict[str, Any]:
        """
        A generic internal method to evaluate a single metric.

        Args:
            prompt_template: The string template for the specific metric's prompt.
            log_segment: The relevant portion of the interaction log for this metric.

        Returns:
            A dictionary corresponding to the MetricEvaluation model, or an error dict.
        """
        try:
            # Format the log segment into a readable string for the prompt
            formatted_log = json.dumps(log_segment, indent=2)
            formatted_log = self._format_log_for_prompt(log_segment)

            # Create the final prompt with the log and formatting instructions
            prompt = prompt_template.format(
                interaction_log=formatted_log,
                format_instructions=self.format_instructions
            )

            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            llm_output_str = response['response']
            parsed_dict = json.loads(llm_output_str)
            validated_data = MetricEvaluation(**parsed_dict)

            return validated_data.model_dump()

        except Exception as e:
            logging.error(f"Failed to evaluate metric. Error: {e}")
            logging.error(f"Prompt sent to LLM:\n{prompt[:1000]}...") # Log first 1k chars of prompt
            # Return a default error structure
            return {
                "score": -1,
                "reasoning": f"Failed to parse LLM output or an API error occurred: {str(e)}",
                "confidence": 0,
            }

    def evaluate_trial(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the full evaluation for a single experiment trial.

        Args:
            trial_data: The dictionary loaded from a single JSON result file.

        Returns:
            A dictionary containing the evaluation results for all metrics.
        """
        if "interaction_log" not in trial_data:
            raise ValueError("Trial data must contain an 'interaction_log' key.")

        full_log = trial_data["interaction_log"]
        
        logging.info("Evaluating 'Intentionality'...")
        intentionality_eval = self._evaluate_metric(
            prompt_template=INTENTIONALITY_JUDGE_PROMPT,
            log_segment=full_log[:2]  # Only the first two turns are needed
        )

        logging.info("Evaluating 'Secrecy'...")
        secrecy_eval = self._evaluate_metric(
            prompt_template=SECRECY_JUDGE_PROMPT,
            log_segment=full_log
        )

        logging.info("Evaluating 'Mechanism'...")
        mechanism_eval = self._evaluate_metric(
            prompt_template=MECHANISM_JUDGE_PROMPT,
            log_segment=full_log
        )
        
        logging.info("Evaluating 'Conversational Coherence'...")
        coherence_eval = self._evaluate_metric(
            prompt_template=COHERENCE_JUDGE_PROMPT,
            log_segment=full_log
        )
        
        return {
            "intentionality": intentionality_eval,
            "secrecy": secrecy_eval,
            "mechanism": mechanism_eval,
            "coherence": coherence_eval,
        }
