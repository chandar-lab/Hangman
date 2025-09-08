import logging
import json
from typing import Dict, Any, List, Optional, Literal

from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from hangman.providers.llmprovider import LLMProvider
from hangman.evaluation.prompt_registry import get_prompts

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class LLMJudge:
    """
    A generic, prompt-driven LLM judge for any supported game.

    - No Pydantic. Parses the model JSON directly with best-effort robustness.
    - Selects prompts via a (game, mode) registry.
    """

    def __init__(
        self,
        judge_llm_provider: LLMProvider,
        game: str,
        mode: Literal["behavioral", "memory", "both"] = "behavioral",
    ) -> None:
        if not isinstance(judge_llm_provider, LLMProvider):
            raise TypeError("judge_llm_provider must be an instance of LLMProvider.")

        self.llm = judge_llm_provider
        self.game = game
        self.mode = mode
        self.parser = PydanticOutputParser(pydantic_object=MetricEvaluation)
        logging.info(
            f"LLMJudge initialized for game='{self.game}', mode='{self.mode}', model={self.llm.config.get('model_name')}"
        )

    def _format_log_for_prompt(self, log_segment: List[List[Any]], include_private: bool) -> str:
        """
        Formats the interaction log into a readable, turn-by-turn string.
        Each turn is [utterance, private_state]. TODO: It strongly assumes that the first mover is the player, make it modular. 
        """
        formatted_parts: List[str] = []
        for i, turn_data in enumerate(log_segment):
            turn_number = i + 1
            # Defensive: tolerate ill-formed items
            utterance = turn_data[0] if isinstance(turn_data, (list, tuple)) and len(turn_data) > 0 else ""
            private_state = turn_data[1] if isinstance(turn_data, (list, tuple)) and len(turn_data) > 1 else ""

            if i % 2 == 0:
                actor = "Player"
                formatted_parts.append(f"--- TURN {turn_number} ({actor}) ---")
                formatted_parts.append("PLAYER UTTERANCE:")
                formatted_parts.append(str(utterance))
            else:
                actor = "Agent"
                formatted_parts.append(f"--- TURN {turn_number} ({actor}) ---")
                formatted_parts.append("AGENT UTTERANCE:")
                formatted_parts.append(str(utterance))
                if include_private and private_state:
                    formatted_parts.append("\nAGENT'S PRIVATE MEMORY:")
                    formatted_parts.append(str(private_state))

            formatted_parts.append("")

        return "\n".join(formatted_parts)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Best-effort JSON extraction from a model response string."""
        # Try direct
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to find the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass

        # Fallback
        return {
            "score": -1,
            "reasoning": "Judge output was not valid JSON.",
            "confidence": 0,
        }

    def _evaluate_metric(
        self,
        prompt_template: str,
        log_segment: List[List[Any]],
        include_private: bool,
        format_instructions: str,
    ) -> Dict[str, Any]:
        formatted_log = self._format_log_for_prompt(log_segment, include_private=include_private)
        prompt = prompt_template.format(
            interaction_log=formatted_log, format_instructions=format_instructions
        )
        try:
            # Send prompt as a HumanMessage to the provider (expected message type)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            llm_output_str = response.get("response", "")
        except Exception as e:
            logging.error(f"Judge model invocation failed: {e}")
            return {
                "score": -1,
                "reasoning": f"Invocation error: {e}",
                "confidence": 0,
            }

        parsed = self._extract_json(llm_output_str)
        validated_data = MetricEvaluation(**parsed)
        return validated_data.model_dump()

    def evaluate_trial(
        self,
        trial_data: Dict[str, Any],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the given trial JSON using the selected (game, mode) prompts.

        metrics: subset of [intentionality, secrecy, mechanism, coherence]. If None, evaluate all.
        """
        if "interaction_log" not in trial_data:
            raise ValueError("Trial data must contain an 'interaction_log' key.")

        full_log: List[List[Any]] = trial_data["interaction_log"]
        chosen_metrics = metrics or ["intentionality", "secrecy", "mechanism", "coherence"]

        def _run_for_mode(effective_mode: Literal["behavioral", "memory"]) -> Dict[str, Any]:
            include_private = (effective_mode == "memory")
            prompt_bundle = get_prompts(self.game, effective_mode)
            metric_prompts = prompt_bundle["metrics"]
            format_instructions = prompt_bundle["format_instructions"]

            results_local: Dict[str, Any] = {}

            if "intentionality" in chosen_metrics and "intentionality" in metric_prompts:
                logging.info(f"Evaluating 'Intentionality' ({effective_mode})...")
                results_local["intentionality"] = self._evaluate_metric(
                    prompt_template=metric_prompts["intentionality"],
                    log_segment=full_log[:2],  # early-turn signal
                    include_private=include_private,
                    format_instructions=format_instructions,
                )

            if "secrecy" in chosen_metrics and "secrecy" in metric_prompts:
                logging.info(f"Evaluating 'Secrecy' ({effective_mode})...")
                results_local["secrecy"] = self._evaluate_metric(
                    prompt_template=metric_prompts["secrecy"],
                    log_segment=full_log,
                    include_private=include_private,
                    format_instructions=format_instructions,
                )

            if "mechanism" in chosen_metrics and "mechanism" in metric_prompts:
                logging.info(f"Evaluating 'Mechanism' ({effective_mode})...")
                results_local["mechanism"] = self._evaluate_metric(
                    prompt_template=metric_prompts["mechanism"],
                    log_segment=full_log,
                    include_private=include_private,
                    format_instructions=format_instructions,
                )

            if "coherence" in chosen_metrics and "coherence" in metric_prompts:
                logging.info(f"Evaluating 'Conversational Coherence' ({effective_mode})...")
                results_local["coherence"] = self._evaluate_metric(
                    prompt_template=metric_prompts["coherence"],
                    log_segment=full_log,
                    include_private=include_private,
                    format_instructions=format_instructions,
                )

            return results_local

        def _run_winner() -> Dict[str, Any]:
            """Behavioral-only winner determination using the dedicated prompt and schema."""
            prompt_bundle = get_prompts(self.game, "behavioral")
            metric_prompts = prompt_bundle.get("metrics", {})
            winner_prompt = metric_prompts.get("winner")
            winner_format = prompt_bundle.get("winner_format_instructions", "")

            if not winner_prompt:
                # Fallback if prompt is missing
                return {
                    "winner": "Unknown",
                    "reasoning": "Winner prompt unavailable.",
                    "confidence": 0,
                }

            formatted_log = self._format_log_for_prompt(full_log, include_private=False)
            prompt = winner_prompt.format(
                interaction_log=formatted_log,
                format_instructions=winner_format,
            )

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                llm_output_str = response.get("response", "")
            except Exception as e:
                logging.error(f"Winner judge model invocation failed: {e}")
                return {
                    "winner": "Unknown",
                    "reasoning": f"Invocation error: {e}",
                    "confidence": 0,
                }

            parsed = self._extract_json(llm_output_str)

            # Minimal validation and normalization
            allowed = {"Player", "Agent", "Unknown"}
            winner_val = parsed.get("winner", "Unknown")
            if not isinstance(winner_val, str) or winner_val not in allowed:
                winner_val = "Unknown"

            reasoning_val = parsed.get("reasoning", "")
            if not isinstance(reasoning_val, str):
                reasoning_val = str(reasoning_val)

            confidence_val = parsed.get("confidence", 0)
            try:
                confidence_val = int(confidence_val)
            except Exception:
                confidence_val = 0
            confidence_val = max(0, min(100, confidence_val))

            return {
                "winner": winner_val,
                "reasoning": reasoning_val,
                "confidence": confidence_val,
            }

        # Winner is always behavioral; compute it once
        winner_result = _run_winner()

        # If the judge is asked to produce both views, return both, and include winner in behavioral
        if self.mode == "both":
            behavioral_results = _run_for_mode("behavioral")
            behavioral_results["winner"] = winner_result
            memory_results = _run_for_mode("memory")
            return {"behavioral": behavioral_results, "memory": memory_results}

        if self.mode == "behavioral":
            behavioral_results = _run_for_mode("behavioral")
            behavioral_results["winner"] = winner_result
            return {"behavioral": behavioral_results}

        # mode == "memory": still include a behavioral section for winner
        memory_results = _run_for_mode("memory")
        return {"memory": memory_results, "behavioral": {"winner": winner_result}}