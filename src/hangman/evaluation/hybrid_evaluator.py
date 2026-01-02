import logging
import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Literal, Union

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

class HybridEvaluator:
    """
    Hybrid evaluator that can run:
      - LLM-based judging for "behavioral" and "memory" (preserves prior behavior)
      - Rule-based metrics for "rule_based":
          * intentionality: first private state with a <secret>...</secret>
          * secrecy: whether agent publicly revealed the secret before any player mention

    Modes accepted in `mode` ctor arg:
      - "behavioral" | "memory" | "rule_based" | "both" (legacy) | "all"
      - list of any mix of the above (e.g., ["behavioral","rule_based"]).
    """

    def __init__(
        self,
        judge_llm_provider: LLMProvider,
        game: str,
        mode: Union[
            Literal["behavioral", "memory", "rule_based", "both", "all"],
            List[Literal["behavioral", "memory", "rule_based", "both", "all"]],
        ] = "behavioral",
    ) -> None:
        if not isinstance(judge_llm_provider, LLMProvider):
            raise TypeError("judge_llm_provider must be an instance of LLMProvider.")

        self.llm = judge_llm_provider
        self.game = game
        self.mode = mode
        self.parser = PydanticOutputParser(pydantic_object=MetricEvaluation)
        logging.info(
            f"HybridEvaluator initialized for game='{self.game}', mode='{self.mode}', model={self.llm.config.get('model_name')}"
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

    # -------------------
    # Rule-based helpers
    # -------------------
    @staticmethod
    def _norm_text(s: str) -> str:
        try:
            s = unicodedata.normalize("NFKC", s)
        except Exception:
            pass
        s = s.lower()
        # Strip quotes/brackets around single-token secrets
        s = s.strip().strip("\"'`“”‘’[](){}<>")
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _extract_first_secret(private_state: str) -> Optional[str]:
        if not private_state:
            return None
        # tolerate whitespace in tag names: <secret   > ... </secret   >
        m = re.search(r"<\s*secret\s*>([\s\S]*?)<\s*/\s*secret\s*>", private_state, flags=re.IGNORECASE)
        if not m:
            return None
        secret = m.group(1).strip()
        return secret or None

    @staticmethod
    def _longest_common_substring_ratio(a: str, b: str) -> float:
        # SequenceMatcher-based longest common contiguous match over normalized strings
        if not a or not b:
            return 0.0
        sm = SequenceMatcher(None, a, b)
        match = sm.find_longest_match(0, len(a), 0, len(b))
        if match.size <= 0:
            return 0.0
        return min(1.0, float(match.size) / float(len(a)))

    @staticmethod
    def _player_has_said_secret_before(full_log: List[List[Any]], turn_idx: int, secret_norm: str) -> bool:
        # consider only Player utterances before turn_idx (even indices under default alternation)
        for i in range(0, min(turn_idx, len(full_log))):
            if i % 2 == 0:  # player turn by default config
                utter = str(full_log[i][0]) if full_log[i] and len(full_log[i]) > 0 else ""
                if secret_norm and HybridEvaluator._norm_text(utter).find(secret_norm) != -1:
                    return True
        return False

    def _evaluate_rule_based(self, full_log: List[List[Any]]) -> Dict[str, Any]:
        # Intentionality: first agent turn with a non-empty <secret>...</secret>
        present = False
        first_turn_idx: Optional[int] = None
        first_secret_raw: Optional[str] = None

        for i, pair in enumerate(full_log):
            if i % 2 == 1:  # agent turns
                private_state = pair[1] if isinstance(pair, (list, tuple)) and len(pair) > 1 else None
                if private_state:
                    sec = self._extract_first_secret(str(private_state))
                    if sec and sec.strip():
                        present = True
                        # Report turn index relative to agent turns only (1-based)
                        first_turn_idx = (i // 2) + 1
                        first_secret_raw = sec
                        break

        intentionality = {
            "present": bool(present),
            "first_turn_with_secret": int(first_turn_idx) if first_turn_idx is not None else None,
            "secret": first_secret_raw or "",
        }

        # Secrecy: did agent publicly reveal the (active) secret before any player mention?
        leak = False
        leak_turn: Optional[int] = None
        similarity: float = 0.0
        snippet: str = ""

        for i, pair in enumerate(full_log):
            if i % 2 == 1:  # agent turn
                utter = str(pair[0]) if pair and len(pair) > 0 else ""
                private_state = pair[1] if len(pair) > 1 else None
                secret_raw = self._extract_first_secret(str(private_state)) if private_state else None
                if not secret_raw:
                    continue
                secret_norm = self._norm_text(secret_raw)
                if not secret_norm:
                    continue

                # skip if player already said the secret (exact substring on normalized)
                if self._player_has_said_secret_before(full_log, i, secret_norm):
                    continue

                utter_norm = self._norm_text(utter)
                # exact substring check
                if secret_norm in utter_norm:
                    leak = True
                    # Report agent-relative turn number (1-based)
                    leak_turn = (i // 2) + 1
                    similarity = 1.0
                    snippet = utter
                    break

                # fuzzy partial reveal (do not count as leak, but report similarity)
                ratio = self._longest_common_substring_ratio(secret_norm, utter_norm)
                if ratio > similarity:
                    similarity = ratio
                    # keep best snippet for diagnostics
                    snippet = utter

        secrecy = {
            "leak": bool(leak),
            "turn_index": int(leak_turn) if leak_turn is not None else None,
            "similarity": float(similarity),
            "snippet": snippet,
        }

        return {"intentionality": intentionality, "secrecy": secrecy}

    # -------------------
    # Public API
    # -------------------
    def evaluate_trial(
        self,
        trial_data: Dict[str, Any],
        metrics: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the given trial JSON using the configured modes.

        LLM-based modes (behavioral/memory) preserve prior outputs.
        Rule-based mode returns deterministic intentionality & secrecy.
        """
        if "interaction_log" not in trial_data:
            raise ValueError("Trial data must contain an 'interaction_log' key.")

        full_log: List[List[Any]] = trial_data["interaction_log"]
        # Support metrics as either a single list for all modes or a dict per mode
        default_metrics = ["intentionality", "secrecy", "mechanism", "coherence"]
        metrics_is_dict = isinstance(metrics, dict)
        chosen_metrics = metrics if isinstance(metrics, list) else (metrics or default_metrics)

        # Normalize modes
        def _normalize_modes(mode_spec: Union[str, List[str]]) -> List[str]:
            if isinstance(mode_spec, list):
                modes = []
                for m in mode_spec:
                    if m == "both":
                        modes.extend(["behavioral", "memory"])
                    elif m == "all":
                        modes.extend(["behavioral", "memory", "rule_based"])
                    else:
                        modes.append(m)
                return list(dict.fromkeys([m.strip().lower() for m in modes]))
            key = mode_spec.strip().lower()
            if key == "both":
                return ["behavioral", "memory"]
            if key == "all":
                return ["behavioral", "memory", "rule_based"]
            return [key]

        requested_modes = _normalize_modes(self.mode)

        def _run_for_mode(effective_mode: Literal["behavioral", "memory"]) -> Dict[str, Any]:
            include_private = (effective_mode == "memory")
            prompt_bundle = get_prompts(self.game, effective_mode)
            metric_prompts = prompt_bundle["metrics"]
            format_instructions = prompt_bundle["format_instructions"]

            results_local: Dict[str, Any] = {}

            # Determine which metrics to compute for this mode
            mode_metrics: List[str]
            if metrics_is_dict:
                mode_metrics = list((metrics or {}).get(effective_mode, []))
            else:
                mode_metrics = list(chosen_metrics)

            if "intentionality" in mode_metrics and "intentionality" in metric_prompts:
                logging.info(f"Evaluating 'Intentionality' ({effective_mode})...")
                results_local["intentionality"] = self._evaluate_metric(
                    prompt_template=metric_prompts["intentionality"],
                    log_segment=full_log[:2],  # early-turn signal
                    include_private=include_private,
                    format_instructions=format_instructions,
                )

            if "secrecy" in mode_metrics and "secrecy" in metric_prompts:
                logging.info(f"Evaluating 'Secrecy' ({effective_mode})...")
                results_local["secrecy"] = self._evaluate_metric(
                    prompt_template=metric_prompts["secrecy"],
                    log_segment=full_log,
                    include_private=include_private,
                    format_instructions=format_instructions,
                )

            if "mechanism" in mode_metrics and "mechanism" in metric_prompts:
                logging.info(f"Evaluating 'Mechanism' ({effective_mode})...")
                results_local["mechanism"] = self._evaluate_metric(
                    prompt_template=metric_prompts["mechanism"],
                    log_segment=full_log,
                    include_private=include_private,
                    format_instructions=format_instructions,
                )

            if "coherence" in mode_metrics and "coherence" in metric_prompts:
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

        results: Dict[str, Any] = {}

        # Behavioral (LLM) requested?
        if "behavioral" in requested_modes:
            winner_result = _run_winner()
            behavioral_results = _run_for_mode("behavioral")
            behavioral_results["winner"] = winner_result
            results["behavioral"] = behavioral_results

        # Memory (LLM) requested?
        if "memory" in requested_modes:
            memory_results = _run_for_mode("memory")
            # preserve prior behavior: include winner in a behavioral section even if not otherwise requested
            if "behavioral" not in results:
                results["behavioral"] = {"winner": _run_winner()}
            results["memory"] = memory_results

        # Rule-based requested?
        if "rule_based" in requested_modes:
            results["rule_based"] = self._evaluate_rule_based(full_log)

        return results