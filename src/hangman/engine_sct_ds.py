import os
import json
import datetime
import re
from typing import Any, Dict, List, Optional
import difflib

from hangman.agents.base_agent import BaseAgent
from hangman.games.base_game import BaseGame
from hangman.games.diagnosis_simulator_sct import DiagnosisSimulatorSCTGame
from hangman.players.deterministic_diagnosis_simulator_player import (
    DeterministicDiagnosisSimulatorPlayer,
)
from hangman.providers.llmprovider import load_llm_provider, LLMProvider
from hangman.sct import diagnosis_utils as dxu
from hangman.evaluation.sct_evaluator import SCTEvaluator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from hangman.prompts.diagnosis_simulator_sct import REVEAL_SECRET_PROMPT, SCT_YES_NO_PROMPT


def _sanitize_branch_id(branch_id: str) -> str:
    """
    Sanitize a branch ID to ensure compatibility with agent naming.
    
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


class DiagnosisSCTController:
    """
    Controller for running a single Diagnosis Simulator Self-Consistency Test episode.

    Flow mirrors the Hangman SCT controller but uses diagnosis-specific prompts,
    deterministic player, and candidate generation.
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

        # ddxplus CSV paths from top-level SCT config (Option B)
        evidences_csv_path = self.sct_cfg.get("evidences_csv_path", "ddxplus/evidences.csv")
        conditions_csv_path = self.sct_cfg.get("conditions_csv_path", "ddxplus/conditions.csv")

        # Deterministic scripted player (trial-aware seed)
        self.player = DeterministicDiagnosisSimulatorPlayer(
            evidences_csv_path=evidences_csv_path,
            conditions_csv_path=conditions_csv_path,
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
        # Similarity config (SequenceMatcher ratio). Default threshold 0.85
        try:
            sim_threshold = float(self.sct_cfg.get("similarity_threshold", 0.85))
        except Exception:
            sim_threshold = 0.85

        def _normalize_label(s: Optional[str]) -> str:
            t = (s or "").strip().lower()
            # keep letters, spaces, hyphens -> then collapse and remove hyphens
            t = "".join(ch if (ch.isalpha() or ch in [" ", "-"]) else " " for ch in t)
            t = t.replace("-", " ")
            t = " ".join(t.split())
            return t

        def _too_similar(a: str, b: str) -> bool:
            if not a or not b:
                return False
            ra = _normalize_label(a)
            rb = _normalize_label(b)
            if ra == rb:
                return True
            return difflib.SequenceMatcher(None, ra, rb).ratio() >= sim_threshold

        def _append_if_novel(acc: List[str], item: Optional[str]) -> None:
            if not item:
                return
            for ex in acc:
                if _too_similar(ex, item):
                    return
            acc.append(item)

        # Summarize WM secrets from private states (agent turns only)
        agent_privates: List[Optional[str]] = []
        for i, pair in enumerate(self.game.get_full_state()):
            if i % 2 == 1:
                agent_privates.append(pair[1] if isinstance(pair, (list, tuple)) and len(pair) > 1 else None)

        wm = dxu.summarize_secret_history(agent_privates)
        last_secret: Optional[str] = wm.get("last_secret")

        n_cand = int(self.sct_cfg.get("n_candidate_secrets", 10))
        method = ((self.sct_cfg.get("stateless_candidates", {}) or {}).get("method") or "deterministic").strip().lower()
        llm = self._load_cand_llm_if_needed()
        llm_max_n = (self.sct_cfg.get("stateless_candidates", {}).get("llm", {}) or {}).get("max_n")

        # Transcript for LLM-based candidate generation
        transcript_text = dxu.format_interaction_log(self.game.get_full_state())

        # Used features from player
        used_features = {}
        try:
            used_features = self.player.get_used_features()
        except Exception:
            used_features = {}

        # Conditions catalog from player
        conditions_catalog = {}
        try:
            conditions_catalog = self.player.get_conditions_catalog()
        except Exception:
            conditions_catalog = {}

        # Reveal secret fork (direct ask)
        revealed_secret: Optional[str] = None
        try:
            from hangman.agents.mem0_agent import Mem0Agent
            from hangman.agents.amem_agent import AMemAgent
            from hangman.agents.lightmem_agent import LightMemAgent
            from hangman.agents.memoryos_agent import MemoryOSAgent
            
            # Build branch messages from pre-fork transcript + reveal prompt
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

            reveal_messages: List[BaseMessage] = list(messages)
            reveal_messages.append(HumanMessage(content=REVEAL_SECRET_PROMPT))
            reveal_out = branch_agent.invoke(reveal_messages)
            reveal_resp = (reveal_out or {}).get("response", "")
            revealed_secret_parsed = dxu.parse_revealed_secret(reveal_resp)
            # Keep the original response for the candidate list (before truncation)
            revealed_secret_original = reveal_resp.strip() if reveal_resp else None

            reveal_branch_private = getattr(branch_agent, "get_private_state", lambda: "")()
            reveal_branch = {
                "interaction_log": [
                    [REVEAL_SECRET_PROMPT, None],
                    [reveal_resp, reveal_branch_private],
                ],
                "answer": reveal_resp,
                "parsed": bool(revealed_secret_parsed is not None),
            }
        except Exception:
            revealed_secret_parsed = None
            revealed_secret_original = None
            reveal_branch = None

        # Base candidates
        base: List[str] = dxu.estimate_candidates_from_transcript(
            transcript_text=transcript_text,
            n=n_cand,
            method=method,
            llm_provider=llm,
            llm_max_n=llm_max_n,
            used_features=used_features if method == "deterministic" else None,
            conditions_catalog=conditions_catalog if method == "deterministic" else None,
        )

        candidates: List[str] = []

        # If deterministic secret extraction failed and an LLM is configured, try LLM-based secret extraction
        if not last_secret and llm is not None:
            try:
                wm_llm = dxu.summarize_secret_history_with_llm(agent_privates, llm_provider=llm)
                last_secret = wm_llm.get("last_secret") or last_secret
            except Exception:
                pass

        # Compose candidates: WM secret -> revealed secret -> base (novel/similarity-aware)
        # Check if last_secret and revealed_secret are essentially the same (containment check)
        # Use parsed version for comparison, but original for candidate list
        last_and_revealed_match = False
        if last_secret and revealed_secret_parsed:
            # If one contains the other (case-insensitive), they're the same secret with different formatting
            if revealed_secret_parsed.lower() in last_secret.lower() or last_secret.lower() in revealed_secret_parsed.lower():
                last_and_revealed_match = True
                # Prefer the original full revealed secret over the parsed/truncated version
                last_secret = revealed_secret_original if revealed_secret_original else revealed_secret_parsed
        
        _append_if_novel(candidates, last_secret)
        # Only add revealed_secret if it's truly different from last_secret
        # Use the original (non-truncated) version for the candidate list
        if not last_and_revealed_match and revealed_secret_original:
            _append_if_novel(candidates, revealed_secret_original)
        for w in base:
            if len(candidates) >= n_cand:
                break
            _append_if_novel(candidates, w)

        # If still short, top-up with LLM-generated candidates
        if len(candidates) < n_cand and llm is not None:
            try:
                need = n_cand - len(candidates)
                pool = dxu.estimate_candidates_from_transcript(
                    transcript_text=transcript_text,
                    n=max(need * 2, need + 5),
                    method="llm",
                    llm_provider=llm,
                    llm_max_n=llm_max_n,
                )
                for w in pool:
                    if len(candidates) >= n_cand:
                        break
                    _append_if_novel(candidates, w)
            except Exception:
                pass

        # Similarity-aware contains/secret_index resolution
        contains_secret = False
        secret_index: Optional[int] = None
        if last_secret:
            for idx, c in enumerate(candidates):
                if _too_similar(c, last_secret):
                    contains_secret = True
                    secret_index = idx
                    break

        return {
            "candidates": candidates,
            "contains_secret": contains_secret,
            "secret_index": secret_index,
            "sct_yes_correct": None,
            "last_secret": last_secret,
            "revealed_secret": revealed_secret_original if revealed_secret_original else revealed_secret_parsed,
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

        # Initial opening
        opening = self.player.opening()
        messages.append(HumanMessage(content=opening))
        self.game.update_state(opening, private_state=None)

        agent_out = self.agent.invoke(messages)
        messages.append(AIMessage(content=agent_out["response"]))
        self.game.update_state(agent_out["response"], private_state=self.agent.get_private_state())
        try:
            self.player.update_with_agent_answer(agent_out.get("response", ""))
        except Exception:
            pass
        turn_pairs += 1

        # Pre-fork turns
        while turn_pairs < t_fork and (2 * turn_pairs) < T_max:
            query = self.player.next_guess()
            messages.append(HumanMessage(content=query))
            self.game.update_state(query, private_state=None)

            agent_out = self.agent.invoke(messages)
            messages.append(AIMessage(content=agent_out["response"]))
            self.game.update_state(agent_out["response"], private_state=self.agent.get_private_state())
            try:
                self.player.update_with_agent_answer(agent_out.get("response", ""))
            except Exception:
                pass
            turn_pairs += 1

        if (2 * turn_pairs) >= T_max and turn_pairs < t_fork:
            safety_reached = True

        pre_fork_messages = list(messages)

        # Build candidate list
        cand_info = self._build_candidates(pre_fork_messages)
        candidates: List[str] = cand_info["candidates"]

        # Branch execution helper
        def _run_branch(word: str) -> Dict[str, Any]:
            from hangman.agents.mem0_agent import Mem0Agent
            from hangman.agents.amem_agent import AMemAgent
            from hangman.agents.lightmem_agent import LightMemAgent
            from hangman.agents.memoryos_agent import MemoryOSAgent
            
            AgentClass = self.agent.__class__
            
            branch_agent: BaseAgent
            
            # Special handling for Mem0Agent - needs memory cloning
            if isinstance(self.agent, Mem0Agent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    branch_id=str(word),
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
            
            # Special handling for LightMemAgent - clone memories and seed window
            elif isinstance(self.agent, LightMemAgent):
                config = self.agent.get_session_config()
                branch_agent = AgentClass(
                    branch_id=_sanitize_branch_id(str(word)),
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
                    branch_id=_sanitize_branch_id(str(word)),
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
                branch_agent = AgentClass(
                    llm_provider=getattr(self.agent, "llm_provider"),
                    strategy=getattr(self.agent, "strategy"),
                )
            elif hasattr(self.agent, "llm_provider"):
                try:
                    branch_agent = AgentClass(llm_provider=getattr(self.agent, "llm_provider"))
                except Exception:
                    branch_agent = AgentClass(llm_provider=getattr(self.agent, "llm_provider"))

            # Seed WM if present
            try:
                pre_state = self.agent.get_state()
                if isinstance(pre_state, dict) and "working_memory" in pre_state:
                    if hasattr(branch_agent, "workflow"):
                        branch_agent.workflow.update_state({"configurable": {"thread_id": "main_thread"}}, {
                            "messages": pre_fork_messages,
                            "working_memory": pre_state.get("working_memory", ""),
                        })
            except Exception:
                pass

            branch_messages: List[BaseMessage] = list(pre_fork_messages)
            prompt = SCT_YES_NO_PROMPT.format(diagnosis=word)
            branch_messages.append(HumanMessage(content=prompt))

            agent_out = branch_agent.invoke(branch_messages)
            resp = (agent_out or {}).get("response", "")

            branch_log = [
                [prompt, None],
                [resp, getattr(branch_agent, "get_private_state", lambda: "")()],
            ]

            a = (resp or "").strip().lower()
            parsed = (a == "yes" or a == "no")
            return {"word": word, "interaction_log": branch_log, "answer": a, "parsed": bool(parsed)}

        branches: List[Dict[str, Any]] = []
        for word in candidates:
            branches.append(_run_branch(word))

        answers: List[Dict[str, Any]] = [{"word": b["word"], "answer": b.get("answer", ""), "parsed": b.get("parsed", False)} for b in branches]

        if cand_info["contains_secret"] and cand_info["secret_index"] is not None and cand_info["secret_index"] < len(answers):
            a = answers[cand_info["secret_index"]]
            cand_info["sct_yes_correct"] = 1 if (a.get("parsed") and a.get("answer") == "yes") else 0

        metadata = {
            "game": getattr(self.game, "name", "diagnosis_simulator_sct"),
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
                    "llm_provider": (self.sct_cfg.get("stateless_candidates", {}).get("llm", {}) or {}).get("provider"),
                },
                "similarity_threshold": float(self.sct_cfg.get("similarity_threshold", 0.85)),
                "evidences_csv_path": self.sct_cfg.get("evidences_csv_path", "ddxplus/evidences.csv"),
                "conditions_csv_path": self.sct_cfg.get("conditions_csv_path", "ddxplus/conditions.csv"),
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
                "safety_reached": False,
            },
        }

        evaluator = SCTEvaluator()
        eval_results = evaluator.evaluate_trial(trial_payload)
        trial_payload["evaluation"] = eval_results

        self._write_log(trial_payload)


