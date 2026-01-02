import os
import sys
import json
import datetime
import re
from typing import Any, Dict, List, Optional
import random

# --- Project Imports ---
from hangman.agents.base_agent import BaseAgent
from hangman.games.base_game import BaseGame
from hangman.games.hangman_sct import HangmanSCTGame
from hangman.players.deterministic_hangman_player import DeterministicHangmanPlayer
from hangman.providers.llmprovider import load_llm_provider, LLMProvider
from hangman.sct import hangman_utils
from hangman.evaluation.sct_evaluator import SCTEvaluator
import hangman.agents as agents_pkg

# LangChain messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def _sanitize_branch_id(branch_id: str) -> str:
    """
    Sanitize a branch ID to ensure compatibility with Letta agent naming.
    
    Letta rejects agent names with special characters like apostrophes.
    Replace non-alphanumeric characters (except spaces and hyphens) with underscores.
    
    Examples:
        "Cushing's Syndrome" -> "Cushing_s_Syndrome"
        "Sjögren's syndrome" -> "Sjogren_s_syndrome"
    """
    if not branch_id:
        return branch_id
    # Replace non-ASCII and special characters with underscores, keep spaces/hyphens/alphanumeric
    sanitized = re.sub(r"[^\w\s\-]", "_", branch_id, flags=re.ASCII)
    return sanitized


class SCTController:
    """
    Controller for running a single Self-Consistency Test episode.

    Orchestrates:
      - Deterministic player pre-fork turns
      - Fork at t_fork and generation of candidate secrets
      - Yes/No queries for each candidate
      - Final evaluation under `evaluation` key
    """

    def __init__(
        self,
        *,
        agent: BaseAgent,
        game: BaseGame,
        results_dir: str,
        sct_cfg: Dict[str, Any],
        providers_config_path: Optional[str] = None,
    ) -> None:
        self.agent = agent
        self.game = game
        self.results_dir = results_dir
        self.sct_cfg = sct_cfg or {}
        self.providers_config_path = providers_config_path
        self.log_filepath: str = ""

        # Deterministic scripted player (trial-aware seed)
        self.player = DeterministicHangmanPlayer(
            random_seed=int(self.sct_cfg.get("random_seed", 1337)),
            t_fork=int(self.sct_cfg.get("t_fork", 6)),
            trial_index=int(self.sct_cfg.get("trial_index", 0)),
        )

        # Optional candidate-generation LLM provider (loaded lazily if needed)
        self._cand_llm: Optional[LLMProvider] = None

    # -----------------
    # Logging helpers
    # -----------------
    def _prepare_log_file(self) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        agent_name = self.agent.__class__.__name__
        self.log_filepath = os.path.join(self.results_dir, f"{agent_name}_{ts}.json")
        print(f"Logging results to: {self.log_filepath}")

    def _write_log(self, payload: Dict[str, Any]) -> None:
        with open(self.log_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)

    # -----------------
    # Candidate helpers
    # -----------------
    def _load_cand_llm_if_needed(self) -> Optional[LLMProvider]:
        try:
            # Load an LLM provider if configured under stateless_candidates.llm.provider,
            # regardless of the candidate generation method. This enables LLM fallback
            # for secret extraction even when method == "deterministic".
            if self._cand_llm is not None:
                return self._cand_llm
            if not self.providers_config_path:
                return None
            llm_name = (self.sct_cfg.get("stateless_candidates", {}).get("llm", {}) or {}).get("provider")
            if not llm_name:
                return None
            self._cand_llm = load_llm_provider(self.providers_config_path, llm_name)
            return self._cand_llm
        except Exception:
            return None

    def _build_candidates(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Returns a dict with keys:
          candidates: List[str]
          contains_secret: bool
          secret_index: Optional[int]
          sct_yes_correct: Optional[int]  # to be computed later after answers; here None
        """
        # Summarize WM secrets from private states
        private_states = [pair[1] if isinstance(pair, (list, tuple)) and len(pair) > 1 else None for pair in self.game.get_full_state() if True]
        # private states are logged for both player and agent; keep agent turns only like evaluator
        agent_privates: List[Optional[str]] = []
        for i, pair in enumerate(self.game.get_full_state()):
            if i % 2 == 1:
                agent_privates.append(pair[1] if len(pair) > 1 else None)

        wm = hangman_utils.summarize_secret_history(agent_privates)
        last_secret: Optional[str] = wm.get("last_secret")

        n_cand = int(self.sct_cfg.get("n_candidate_secrets", 10))
        method = ((self.sct_cfg.get("stateless_candidates", {}) or {}).get("method") or "deterministic").strip().lower()
        dict_path = (self.sct_cfg.get("stateless_candidates", {}).get("deterministic", {}) or {}).get("dictionary_path")
        llm = self._load_cand_llm_if_needed()
        llm_max_n = (self.sct_cfg.get("stateless_candidates", {}).get("llm", {}) or {}).get("max_n")

        # Build transcript text for candidate generation
        transcript_text = hangman_utils.format_interaction_log(self.game.get_full_state())
        # Prefer used letters directly from the deterministic player
        used_letters_from_player = set()
        try:
            if hasattr(self.player, "get_used_letters"):
                used_letters_from_player = set(self.player.get_used_letters())
        except Exception:
            used_letters_from_player = set()

        # --- Reveal secret fork (direct ask) ---
        revealed_secret: Optional[str] = None
        try:
            from hangman.prompts.hangman_sct import REVEAL_SECRET_PROMPT
            from hangman.agents.mem0_agent import Mem0Agent
            from hangman.agents.amem_agent import AMemAgent
            from hangman.agents.letta_agent import LettaAgent
            from hangman.agents.lightmem_agent import LightMemAgent
            from hangman.agents.memoryos_agent import MemoryOSAgent

            # Recreate agent instance of the same class with same providers/strategy
            AgentClass = self.agent.__class__
            branch_agent: BaseAgent
            
            # Special handling for Mem0Agent - needs memory cloning
            if isinstance(self.agent, Mem0Agent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    branch_id="reveal",
                    parent_session_id=config["session_id"],
                    **config
                )
                # Clone pre-fork memories to the reveal branch
                branch_agent.clone_memories_from(self.agent.user_bucket)
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
            
            # Special handling for AMemAgent - clone memories and seed window
            elif isinstance(self.agent, AMemAgent):
                config = self.agent.get_session_config()
                # Extract parent_amem before creating instance (don't pass to constructor)
                parent_amem = config.pop("parent_amem", None)
                
                # Create fresh branch agent instance
                branch_agent = AgentClass(**config)
                
                # Clone memories from parent
                if parent_amem is not None:
                    branch_agent.clone_memories_from(parent_amem)
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
            
            # Special handling for LettaAgent - clone via export/import
            elif isinstance(self.agent, LettaAgent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    llm_provider=config["llm_provider"],
                    letta_config=config["letta_config"],
                    letta_base_url=config["letta_base_url"],
                    session_id=config["session_id"],
                    branch_id="reveal",
                    timeout=config["timeout"]
                )
                # Clone memories and conversation history from parent
                branch_agent.clone_memories_from(config["parent_letta_agent_id"])
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
            
            # Special handling for LightMemAgent - clone memories and seed window
            elif isinstance(self.agent, LightMemAgent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    branch_id="reveal",
                    parent_session_id=config["session_id"],
                    **config
                )
                # Clone pre-fork memories to the reveal branch (pass parent's LightMem instance)
                branch_agent.clone_memories_from(self.agent.lightmem)
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
            
            # Special handling for MemoryOSAgent - clone memories and seed window
            elif isinstance(self.agent, MemoryOSAgent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    branch_id="reveal",
                    parent_session_id=config["session_id"],
                    **config
                )
                # Clone pre-fork memories to the reveal branch (pass parent's MemoryOS instance)
                branch_agent.clone_memories_from(self.agent.memo)
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
                    
            elif hasattr(self.agent, "responder_llm") and hasattr(self.agent, "updater_llm") and hasattr(self.agent, "strategy"):
                branch_agent = AgentClass(
                    llm_provider=getattr(self.agent, "responder_llm"),
                    updater_llm_provider=getattr(self.agent, "updater_llm"),
                    strategy=getattr(self.agent, "strategy"),
                )
            elif hasattr(self.agent, "llm_provider") and hasattr(self.agent, "strategy"):
                branch_agent = AgentClass(
                    llm_provider=getattr(self.agent, "llm_provider"),
                    strategy=getattr(self.agent, "strategy"),
                )
            elif hasattr(self.agent, "llm_provider"):
                try:
                    branch_agent = AgentClass(llm_provider=getattr(self.agent, "llm_provider"))
                except Exception:
                    branch_agent = AgentClass(llm_provider=getattr(self.agent, "llm_provider"))

            # Seed reveal branch agent working memory if WM present
            try:
                pre_state = self.agent.get_state()
                if isinstance(pre_state, dict) and "working_memory" in pre_state:
                    if hasattr(branch_agent, "workflow"):
                        branch_agent.workflow.update_state({"configurable": {"thread_id": "main_thread"}}, {
                            "messages": pre_state.get("messages", []),
                            "working_memory": pre_state.get("working_memory", ""),
                        })
            except Exception:
                pass

            # Build branch messages from pre-fork transcript + reveal prompt
            reveal_messages: List[BaseMessage] = list(messages)
            reveal_messages.append(HumanMessage(content=REVEAL_SECRET_PROMPT))
            reveal_out = branch_agent.invoke(reveal_messages)
            reveal_resp = (reveal_out or {}).get("response", "")
            revealed_secret = hangman_utils.parse_revealed_secret(reveal_resp)

            # Build a reveal branch log similar to candidate branches
            reveal_branch_private = getattr(branch_agent, "get_private_state", lambda: "")()
            reveal_branch = {
                "interaction_log": [
                    [REVEAL_SECRET_PROMPT, None],
                    [reveal_resp, reveal_branch_private],
                ],
                "answer": reveal_resp,
                "parsed": bool(revealed_secret is not None),
            }
        except Exception:
            revealed_secret = None
            reveal_branch = None

        # Base candidates from configured method (LLM recommended)
        base: List[str] = hangman_utils.estimate_candidates_from_transcript(
            transcript_text=transcript_text,
            n=n_cand,
            method=method,
            dictionary_path=dict_path,
            llm_provider=llm,
            llm_max_n=llm_max_n,
            used_letters=used_letters_from_player,
        )

        rng = random.Random(int(self.sct_cfg.get("random_seed", 1337)))
        candidates: List[str] = []

        # If deterministic secret extraction failed and an LLM is configured, try LLM-based secret extraction
        if not last_secret and llm is not None:
            try:
                wm_llm = hangman_utils.summarize_secret_history_with_llm(agent_privates, llm_provider=llm)
                last_secret = wm_llm.get("last_secret") or last_secret
            except Exception:
                pass

        # Compose candidates: WM secret -> revealed secret -> base (dedup), then top up if needed
        # Check if last_secret and revealed_secret are essentially the same (containment check)
        last_and_revealed_match = False
        if last_secret and revealed_secret:
            # If one contains the other (case-insensitive), they're the same secret with different formatting
            if revealed_secret.lower() in last_secret.lower() or last_secret.lower() in revealed_secret.lower():
                last_and_revealed_match = True
                # Prefer the shorter, cleaner one (usually revealed_secret)
                if len(revealed_secret) <= len(last_secret):
                    last_secret = revealed_secret
                # else: keep last_secret as is
        
        # Add last_secret if present
        if last_secret:
            candidates.append(last_secret)
        # Add revealed_secret only if different from last_secret
        if revealed_secret and not last_and_revealed_match and revealed_secret not in candidates:
            candidates.append(revealed_secret)
        for w in base:
            if len(candidates) >= n_cand:
                break
            if w not in candidates:
                candidates.append(w)
        if len(candidates) < n_cand:
            extra2 = hangman_utils.estimate_candidates_from_transcript(
                transcript_text=transcript_text,
                n=max(n_cand * 2, n_cand + 5),
                method=method,
                dictionary_path=dict_path,
                llm_provider=llm,
                llm_max_n=llm_max_n,
                used_letters=used_letters_from_player,
            )
            for w in extra2:
                if len(candidates) >= n_cand:
                    break
                if w not in candidates:
                    candidates.append(w)
        if len(candidates) < n_cand:
            fallback_words = [
                "alpha","bravo","delta","omega","puzzle","python","elephant","computer",
                "science","network","keyboard","monitor","window","garden","rocket","planet"
            ]
            for w in fallback_words:
                if len(candidates) >= n_cand:
                    break
                if w not in candidates:
                    candidates.append(w)

        contains_secret = bool(last_secret in candidates) if last_secret else False
        secret_index = (candidates.index(last_secret) if contains_secret else None) if last_secret else None

        return {
            "candidates": candidates,
            "contains_secret": contains_secret,
            "secret_index": secret_index,
            "sct_yes_correct": None,
            "last_secret": last_secret,
            "revealed_secret": revealed_secret,
            "reveal_branch": reveal_branch,
        }

    # -----------------
    # Run
    # -----------------
    def run(self) -> None:
        # Reset state
        self.agent.reset()
        self.player.reset()
        self.game.reset()

        # Prepare logging
        self._prepare_log_file()

        t_fork = int(self.sct_cfg.get("t_fork", 6))
        T_max = int(self.sct_cfg.get("T_max", max(2 * t_fork, 20)))

        messages: List[BaseMessage] = []
        turn_pairs = 0
        safety_reached = False

        # Propagate trial index to player RNG (if present)
        try:
            if "trial_index" in self.sct_cfg:
                os.environ["SCT_TRIAL_INDEX"] = str(int(self.sct_cfg["trial_index"]))
        except Exception:
            pass

        # Initial opening
        opening = self.player.opening()
        messages.append(HumanMessage(content=opening))
        self.game.update_state(opening, private_state=None)

        agent_out = self.agent.invoke(messages)
        messages.append(AIMessage(content=agent_out["response"]))
        self.game.update_state(agent_out["response"], private_state=self.agent.get_private_state())
        turn_pairs += 1

        # Pre-fork turns
        while turn_pairs < t_fork and (2 * turn_pairs) < T_max:
            guess = self.player.next_guess()
            messages.append(HumanMessage(content=guess))
            self.game.update_state(guess, private_state=None)

            agent_out = self.agent.invoke(messages)
            messages.append(AIMessage(content=agent_out["response"]))
            self.game.update_state(agent_out["response"], private_state=self.agent.get_private_state())
            turn_pairs += 1

        if (2 * turn_pairs) >= T_max and turn_pairs < t_fork:
            safety_reached = True

        # Capture pre-fork state and messages (stop logging to main interaction_log here)
        pre_fork_messages = list(messages)

        # Build candidate list from pre-fork transcript
        cand_info = self._build_candidates(pre_fork_messages)
        candidates: List[str] = cand_info["candidates"]

        # Branch execution helper: re-instantiate agent per candidate and run a single yes/no turn
        branch_counter = [0]  # Mutable counter for branch IDs
        
        def _run_branch(word: str) -> Dict[str, Any]:
            from hangman.prompts.hangman_sct import SCT_YES_NO_PROMPT
            from hangman.agents.mem0_agent import Mem0Agent
            from hangman.agents.amem_agent import AMemAgent
            from hangman.agents.letta_agent import LettaAgent
            from hangman.agents.lightmem_agent import LightMemAgent
            from hangman.agents.memoryos_agent import MemoryOSAgent

            # Recreate agent instance of the same class with same providers/strategy
            AgentClass = self.agent.__class__
            branch_id = branch_counter[0]
            branch_counter[0] += 1

            # Detect constructor signature heuristically
            branch_agent: BaseAgent
            
            # Special handling for Mem0Agent - needs memory cloning
            if isinstance(self.agent, Mem0Agent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    branch_id=str(branch_id),
                    parent_session_id=config["session_id"],
                    **config
                )
                # Clone pre-fork memories to the branch
                branch_agent.clone_memories_from(self.agent.user_bucket)
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
            
            # Special handling for AMemAgent - clone memories and seed window
            elif isinstance(self.agent, AMemAgent):
                config = self.agent.get_session_config()
                # Extract parent_amem before creating instance (don't pass to constructor)
                parent_amem = config.pop("parent_amem", None)
                
                # Create fresh branch agent instance
                branch_agent = AgentClass(**config)
                
                # Clone memories from parent
                if parent_amem is not None:
                    branch_agent.clone_memories_from(parent_amem)
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
            
            # Special handling for LettaAgent - clone via export/import
            elif isinstance(self.agent, LettaAgent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    llm_provider=config["llm_provider"],
                    letta_config=config["letta_config"],
                    letta_base_url=config["letta_base_url"],
                    session_id=config["session_id"],
                    branch_id=_sanitize_branch_id(str(branch_id)),
                    timeout=config["timeout"]
                )
                # Clone memories and conversation history from parent
                branch_agent.clone_memories_from(config["parent_letta_agent_id"])
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
            
            # Special handling for LightMemAgent - clone memories and seed window
            elif isinstance(self.agent, LightMemAgent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    branch_id=str(branch_id),
                    parent_session_id=config["session_id"],
                    **config
                )
                # Clone pre-fork memories to the branch (pass parent's LightMem instance)
                branch_agent.clone_memories_from(self.agent.lightmem)
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
            
            # Special handling for MemoryOSAgent - clone memories and seed window
            elif isinstance(self.agent, MemoryOSAgent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    branch_id=str(branch_id),
                    parent_session_id=config["session_id"],
                    **config
                )
                # Clone pre-fork memories to the branch (pass parent's MemoryOS instance)
                branch_agent.clone_memories_from(self.agent.memo)
                
                # Seed the sliding window with pre-fork messages
                pre_fork_window = self.agent.get_sliding_window_state()
                for msg in pre_fork_window:
                    branch_agent._window.append(msg)
                    
            elif hasattr(self.agent, "responder_llm") and hasattr(self.agent, "updater_llm") and hasattr(self.agent, "strategy"):
                branch_agent = AgentClass(
                    llm_provider=getattr(self.agent, "responder_llm"),
                    updater_llm_provider=getattr(self.agent, "updater_llm"),
                    strategy=getattr(self.agent, "strategy"),
                )
            elif hasattr(self.agent, "llm_provider") and hasattr(self.agent, "strategy"):
                # ReActMemAgent
                branch_agent = AgentClass(
                    llm_provider=getattr(self.agent, "llm_provider"),
                    strategy=getattr(self.agent, "strategy"),
                )
            elif hasattr(self.agent, "llm_provider"):
                # Vanilla/Public/PrivateCoT fallbacks
                try:
                    branch_agent = AgentClass(llm_provider=getattr(self.agent, "llm_provider"))
                except Exception:
                    branch_agent = AgentClass(llm_provider=getattr(self.agent, "llm_provider"))

            # Seed branch agent working memory if WM present
            try:
                pre_state = self.agent.get_state()
                if isinstance(pre_state, dict) and "working_memory" in pre_state:
                    # update branch main thread directly
                    if hasattr(branch_agent, "workflow"):
                        branch_agent.workflow.update_state({"configurable": {"thread_id": "main_thread"}}, {
                            "messages": pre_state.get("messages", []),
                            "working_memory": pre_state.get("working_memory", ""),
                        })
            except Exception:
                pass

            # Branch-local messages = pre-fork + question
            branch_messages: List[BaseMessage] = list(pre_fork_messages)
            prompt = SCT_YES_NO_PROMPT.format(word=word)
            branch_messages.append(HumanMessage(content=prompt))

            agent_out = branch_agent.invoke(branch_messages)
            resp = (agent_out or {}).get("response", "")

            # Compose branch interaction log (question + answer + private state)
            branch_log = [
                [prompt, None],
                [resp, getattr(branch_agent, "get_private_state", lambda: "")()],
            ]

            a = (resp or "").strip().lower()
            parsed = (a == "yes" or a == "no")
            return {"word": word, "interaction_log": branch_log, "answer": a, "parsed": bool(parsed)}

        # Execute branches independently
        branches: List[Dict[str, Any]] = []
        for word in candidates:
            branches.append(_run_branch(word))

        # Compute answers array from branches for evaluator compatibility
        answers: List[Dict[str, Any]] = [{"word": b["word"], "answer": b.get("answer", ""), "parsed": b.get("parsed", False)} for b in branches]

        # Compute sct_yes_correct now that answers are known
        if cand_info["contains_secret"] and cand_info["secret_index"] is not None and cand_info["secret_index"] < len(answers):
            a = answers[cand_info["secret_index"]]
            cand_info["sct_yes_correct"] = 1 if (a.get("parsed") and a.get("answer") == "yes") else 0

        # Assemble log payload
        metadata = {
            "game": getattr(self.game, "name", "hangman_sct"),
            "agent_class": self.agent.__class__.__name__,
            "player_class": self.player.__class__.__name__,
            "agent_llm": getattr(self.agent.llm_provider, "config", {}),
            "max_turns": T_max,
            "timestamp": datetime.datetime.now().isoformat(),
            "sct": {
                "t_fork": t_fork,
                "T_max": T_max,
                "random_seed": int(self.sct_cfg.get("random_seed", 1337)),
                "n_candidate_secrets": int(self.sct_cfg.get("n_candidate_secrets", 10)),
                "candidate_generation": {
                    "method": (self.sct_cfg.get("stateless_candidates", {}).get("method") or "deterministic"),
                    "dictionary_path": (self.sct_cfg.get("stateless_candidates", {}).get("deterministic", {}) or {}).get("dictionary_path"),
                    "llm_provider": (self.sct_cfg.get("stateless_candidates", {}).get("llm", {}) or {}).get("provider"),
                },
            },
        }

        trial_payload: Dict[str, Any] = {
            "metadata": metadata,
            "interaction_log": self.game.get_full_state(),
            "sct": {
                "t_fork": t_fork,
                "candidates": candidates,
                "answers": answers,
                "branches": branches,
                "ground_truth_secret": cand_info.get("last_secret"),
                "revealed_secret": cand_info.get("revealed_secret"),
                "reveal_branch": cand_info.get("reveal_branch"),
                "contains_secret": bool(cand_info["contains_secret"]),
                "secret_index": cand_info["secret_index"],
                "sct_yes_correct": cand_info["sct_yes_correct"],
                "safety_reached": safety_reached,
            },
        }

        # Evaluate
        evaluator = SCTEvaluator()
        eval_results = evaluator.evaluate_trial(trial_payload)
        trial_payload["evaluation"] = eval_results

        # Add LLM usage tracking
        from hangman.tracker import get_current_tracker
        tracker = get_current_tracker()
        if tracker is not None:
            trial_payload["llm_usage"] = tracker.to_dict()
        else:
            # Fallback if tracker not initialized (shouldn't happen in normal flow)
            trial_payload["llm_usage"] = None

        # Persist
        self._write_log(trial_payload)


if __name__ == "__main__":
    print("This module provides SCTController; use run_experiment_sct.py to launch experiments.")


