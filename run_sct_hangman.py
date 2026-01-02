import sys
import os

# CRITICAL: Patch OpenAI SDK BEFORE any agent/provider imports
from hangman.tracker import patch_all
patch_all()

# Now safe to import everything else
import yaml
import argparse
import inspect
from datetime import datetime
from typing import Any, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import cycle
from collections.abc import Mapping

# Project imports
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.games import create_game
from hangman.engine_sct_hangman import SCTController

# Agents package for dynamic class resolution
import hangman.agents as agents_pkg
def _to_plain(obj: Any) -> Any:
    """Return obj (placeholder to keep call sites; YAML already returns plain types)."""
    return obj



def _instantiate_agent_from_spec(
    agent_spec: Any,
    providers_config_path: str,
    default_main: LLMProvider,
):
    """
    Create an agent instance from a dict like {"ClassName": {kwargs...}}.
    Resolves any *llm_provider kwargs from provider names to LLMProvider objects.
    Supplies sensible defaults for constructor params when omitted.
    """
    if isinstance(agent_spec, str):
        class_name = agent_spec
        raw_kwargs = {}
    elif isinstance(agent_spec, Mapping) and len(agent_spec) == 1:
        class_name, raw_kwargs = next(iter(agent_spec.items()))
        raw_kwargs = _to_plain(raw_kwargs) or {}
    else:
        raise ValueError(f"Invalid agent specification: {agent_spec}")

    if not hasattr(agents_pkg, class_name):
        raise ValueError(f"Unknown agent class '{class_name}' in run config.")
    AgentClass = getattr(agents_pkg, class_name)

    # Resolve any provider-name strings to LLMProvider objects
    kwargs: Dict[str, Any] = {}
    for k, v in raw_kwargs.items():
        if k.endswith("_llm_provider") and isinstance(v, str):
            kwargs[k] = load_llm_provider(providers_config_path, v)
        else:
            kwargs[k] = v

    # Provide defaults for common constructor args
    sig = inspect.signature(AgentClass.__init__)
    params = sig.parameters
    if "llm_provider" in params and "llm_provider" not in kwargs:
        kwargs["llm_provider"] = default_main
    if "main_llm_provider" in params and "main_llm_provider" not in kwargs:
        kwargs["main_llm_provider"] = default_main
    if "responder_llm_provider" in params and "responder_llm_provider" not in kwargs:
        kwargs["responder_llm_provider"] = default_main
    if "updater_llm_provider" in params and "updater_llm_provider" not in kwargs:
        kwargs["updater_llm_provider"] = default_main
    
    # Default handling for Mem0Agent config path
    if "mem0_config_path" in params and "mem0_config_path" not in kwargs:
        default_mem0_config = "./config/mem0_config.yaml"
        if os.path.exists(default_mem0_config):
            kwargs["mem0_config_path"] = default_mem0_config
    
    # Default handling for LightMemAgent config path
    if "lightmem_config_path" in params and "lightmem_config_path" not in kwargs:
        default_lightmem_config = "./config/lightmem_config_gptoss_20b.yaml"
        if os.path.exists(default_lightmem_config):
            kwargs["lightmem_config_path"] = default_lightmem_config

    # Drop cosmetic name to avoid passing to ctor if present
    if "name" in kwargs:
        kwargs.pop("name")

    return AgentClass(**kwargs)


def _is_complete_log(filepath: str) -> bool:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        is_dict = isinstance(data, dict)
        has_eval = isinstance(data.get("evaluation"), dict)
        has_interaction_log = isinstance(data.get("interaction_log", []), list)
        responses = [x[0] for x in data.get("interaction_log", [])]
        has_errors = any([isinstance(r, str) and r.startswith("Error: ") for r in responses])
        return is_dict and has_eval and has_interaction_log and not has_errors
    except Exception:
        return False


def _run_trial_job(
    *,
    providers_config_path: str,
    agent_spec: Any,
    main_provider_name: str,
    game_name: str,
    sct_cfg: Dict[str, Any],
    results_dir: str,
    trial_index: int,
    session_prefix: str,
) -> bool:
    try:
        main_llm = load_llm_provider(providers_config_path, main_provider_name)
        
        # Build trial-specific session_id for memory-based agents
        # CRITICAL: Must be set BEFORE instantiation so LightMem uses correct collection name
        trial_session_id = f"{session_prefix}_trial_{trial_index}"
        
        # Inject session_id into agent kwargs if needed
        agent_spec_with_session = agent_spec
        if isinstance(agent_spec, dict) and len(agent_spec) == 1:
            class_name, raw_kwargs = next(iter(agent_spec.items()))
            # Only inject for agents that support session_id parameter
            if class_name in ["Mem0Agent", "AMemAgent", "LightMemAgent", "MemoryOSAgent"]:
                raw_kwargs = dict(raw_kwargs or {})
                raw_kwargs["session_id"] = trial_session_id
                agent_spec_with_session = {class_name: raw_kwargs}
        
        agent = _instantiate_agent_from_spec(
            agent_spec=agent_spec_with_session,
            providers_config_path=providers_config_path,
            default_main=main_llm,
        )
        
        # Post-instantiation updates for specific agent types
        from hangman.agents.mem0_agent import Mem0Agent
        from hangman.agents.amem_agent import AMemAgent
        from hangman.agents.lightmem_agent import LightMemAgent
        from hangman.agents.memoryos_agent import MemoryOSAgent
        
        if isinstance(agent, Mem0Agent):
            # Mem0Agent: also update user_bucket and _thread_id
            agent.user_bucket = f"{agent.session_id}__user"
            agent._thread_id = f"mem0_main__{agent.session_id}"
        elif isinstance(agent, LightMemAgent):
            # LightMemAgent: also update _thread_id
            agent._thread_id = f"lightmem_main__{agent.session_id}"
        elif isinstance(agent, MemoryOSAgent):
            # MemoryOSAgent: also update _thread_id
            agent._thread_id = f"memoryos_main__{agent.session_id}"

        game, _ = create_game(game_name)
        sct_cfg_local = dict(sct_cfg or {})
        sct_cfg_local["trial_index"] = int(trial_index)
        controller = SCTController(
            agent=agent,
            game=game,
            results_dir=results_dir,
            sct_cfg=sct_cfg_local,
            providers_config_path=providers_config_path,
        )
        controller.run()
        return True
    except Exception:
        return False


def run_experiments_sct(
    run_config_path: str = "./config/hangman_sct_run.yaml",
    providers_config_path: str = "./config/config.yaml",
):
    # Load run config
    try:
        with open(run_config_path, "r") as f:
            run_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"❌ Run config not found at '{run_config_path}'.")
        sys.exit(1)

    # Normalize to plain Python types for robust isinstance checks
    run_cfg = _to_plain(run_cfg) or {}

    game_name = run_cfg.get("game", "hangman_sct")
    agents_to_test = run_cfg.get("agents", []) or []
    num_trials = int(run_cfg.get("num_trials", 10))
    base_results_dir = run_cfg.get("results_dir", f"results/{game_name}")

    # Providers
    providers = run_cfg.get("providers", {}) or {}
    MAIN_LLM_NAME = providers.get("main", "qwen3_14b_vllm_hermes")
    MAIN_POOL = providers.get("main_pool") or []

    def _to_int(value: Any, default: int) -> int:
        try:
            iv = int(value)
            return iv if iv > 0 else default
        except Exception:
            return default

    concurrency = _to_int(providers.get("concurrency", 0), 0)
    if not MAIN_POOL and concurrency > 0:
        MAIN_POOL = [MAIN_LLM_NAME for _ in range(concurrency)]

    # Shared provider objects (sequential path)
    if not MAIN_POOL:
        try:
            main_llm_default = load_llm_provider(providers_config_path, MAIN_LLM_NAME)
            print("✅ LLM Provider initialized.")
        except Exception as e:
            print(f"❌ Failed to initialize LLM Provider: {e}.")
            sys.exit(1)

    # Create the SCT game once to validate routing
    try:
        _, normalized_game = create_game(game_name)
        if normalized_game != "hangman_sct":
            print(f"⚠️ Normalized game name is '{normalized_game}', expected 'hangman_sct'.")
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    # SCT config block
    sct_cfg = run_cfg.get("sct", {}) or {}

    print("--- 🧪 Initializing SCT Experiment ---")

    for agent_spec in agents_to_test:
        # Resolve user-friendly agent name for folder
        if isinstance(agent_spec, str):
            agent_name = agent_spec
        elif isinstance(agent_spec, Mapping) and len(agent_spec) == 1:
            key, val = next(iter(agent_spec.items()))
            if isinstance(val, Mapping) and "name" in val:
                agent_name = val["name"]
            else:
                agent_name = key
        else:
            print(f"❌ Invalid agent spec in config: {agent_spec}. Skipping.")
            continue

        print(f"\n{'='*25} Starting SCT for: {agent_name} {'='*25}")

        agent_results_dir = os.path.join(base_results_dir, agent_name)
        os.makedirs(agent_results_dir, exist_ok=True)

        # Cleanup incomplete logs
        try:
            for fname in os.listdir(agent_results_dir):
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(agent_results_dir, fname)
                if not _is_complete_log(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

        # Count completed trials
        try:
            completed_trials = 0
            for f in os.listdir(agent_results_dir):
                if not f.endswith('.json'):
                    continue
                fpath = os.path.join(agent_results_dir, f)
                try:
                    with open(fpath, "r", encoding="utf-8") as fp:
                        data = yaml.safe_load(fp) or {}
                    if isinstance(data, dict) and isinstance(data.get("evaluation"), dict):
                        completed_trials += 1
                except Exception:
                    continue
        except FileNotFoundError:
            completed_trials = 0

        if completed_trials >= num_trials:
            print(f"✅ Agent {agent_name} already has {completed_trials} trials completed. Skipping.")
            continue

        print(f"▶️  Found {completed_trials} existing trials. Starting from trial {completed_trials + 1}.")

        needed_trials = num_trials - completed_trials

        if MAIN_POOL:
            main_cycle = cycle(MAIN_POOL)
            max_workers = len(MAIN_POOL)
            print(f"▶️  Parallel execution enabled with {max_workers} workers and provider pool: {MAIN_POOL}")

            # Compute session prefix for this agent to prevent memory leakage across runs
            session_prefix = f"{MAIN_LLM_NAME}_{game_name}_{agent_name}"
            
            jobs = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for idx in range(needed_trials):
                    trial_index = completed_trials + idx + 1
                    jobs.append(
                        executor.submit(
                            _run_trial_job,
                            providers_config_path=providers_config_path,
                            agent_spec=agent_spec,
                            main_provider_name=next(main_cycle),
                            game_name="hangman_sct",
                            sct_cfg=sct_cfg,
                            results_dir=agent_results_dir,
                            trial_index=trial_index,
                            session_prefix=session_prefix,
                        )
                    )

                for _ in as_completed(jobs):
                    pass
        else:
            # Compute session prefix for this agent to prevent memory leakage across runs
            session_prefix = f"{MAIN_LLM_NAME}_{game_name}_{agent_name}"
            
            # Import tracker utilities for per-trial usage tracking
            from hangman.tracker import LLMUsageTracker, set_current_tracker, clear_current_tracker
            
            for idx in range(needed_trials):
                # Create tracker for this trial
                tracker = LLMUsageTracker()
                set_current_tracker(tracker)
                
                try:
                    trial_index = completed_trials + idx + 1
                    
                    # Build trial-specific session_id for memory-based agents
                    # CRITICAL: Must be set BEFORE instantiation so LightMem uses correct collection name
                    trial_session_id = f"{session_prefix}_trial_{trial_index}"
                    
                    # Inject session_id into agent kwargs if needed
                    agent_spec_with_session = agent_spec
                    if isinstance(agent_spec, dict) and len(agent_spec) == 1:
                        class_name, raw_kwargs = next(iter(agent_spec.items()))
                        # Only inject for agents that support session_id parameter
                        if class_name in ["Mem0Agent", "AMemAgent", "LightMemAgent", "MemoryOSAgent"]:
                            raw_kwargs = dict(raw_kwargs or {})
                            raw_kwargs["session_id"] = trial_session_id
                            agent_spec_with_session = {class_name: raw_kwargs}
                    
                    agent = _instantiate_agent_from_spec(
                        agent_spec=agent_spec_with_session,
                        providers_config_path=providers_config_path,
                        default_main=main_llm_default,
                    )
                    
                    # Post-instantiation updates for specific agent types
                    from hangman.agents.mem0_agent import Mem0Agent
                    from hangman.agents.amem_agent import AMemAgent
                    from hangman.agents.lightmem_agent import LightMemAgent
                    from hangman.agents.memoryos_agent import MemoryOSAgent
                    
                    if isinstance(agent, Mem0Agent):
                        # Mem0Agent: also update user_bucket and _thread_id
                        agent.user_bucket = f"{agent.session_id}__user"
                        agent._thread_id = f"mem0_main__{agent.session_id}"
                    elif isinstance(agent, LightMemAgent):
                        # LightMemAgent: also update _thread_id
                        agent._thread_id = f"lightmem_main__{agent.session_id}"
                    elif isinstance(agent, MemoryOSAgent):
                        # MemoryOSAgent: also update _thread_id
                        agent._thread_id = f"memoryos_main__{agent.session_id}"
                    
                    game, _ = create_game("hangman_sct")
                    sct_cfg_local = dict(sct_cfg or {})
                    sct_cfg_local["trial_index"] = int(trial_index)
                    controller = SCTController(
                        agent=agent,
                        game=game,
                        results_dir=agent_results_dir,
                        sct_cfg=sct_cfg_local,
                        providers_config_path=providers_config_path,
                    )
                    controller.run()
                except Exception as e:
                    print(f"--- ❌ ERROR for {agent_name}: {e} ---")
                    continue
                finally:
                    # Always clear tracker after trial (success or failure)
                    clear_current_tracker()

    print(f"\n{'='*30} ✅ SCT Experiments Finished {'='*30}")
    print(f"Results saved in: {base_results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hangman SCT experiments from a YAML run config")
    parser.add_argument(
        "--run-config",
        "-r",
        default=os.environ.get("RUN_CONFIG", "./config/hangman_sct_gptoss_run.yaml"),
        help="Path to the YAML run configuration file (default: ./config/hangman_sct_run.yaml)",
    )
    parser.add_argument(
        "--providers-config",
        "-p",
        default="./config/config.yaml",
        help="Path to providers config YAML used to resolve LLM providers (default: ./config/config.yaml)",
    )
    args = parser.parse_args()

    run_experiments_sct(run_config_path=args.run_config, providers_config_path=args.providers_config)


