#!/usr/bin/env python3
import os
import sys
import json
import glob
import argparse
from typing import Any, Dict, List, Optional

import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import cycle

# Project imports
from hangman.providers.llmprovider import load_llm_provider
from hangman.evaluation.hybrid_evaluator import HybridEvaluator


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_mode(mode: Any) -> Any:
    # Pass-through; HybridEvaluator accepts str or list for mode
    # Keep it as-is from config
    return mode


def _needs_reeval(data: Dict[str, Any]) -> bool:
    eval_block = data.get("evaluation")
    if not isinstance(eval_block, dict):
        return True
    results = eval_block.get("results")
    if not isinstance(results, dict):
        return True
    # User-specified rule: rerun if there is an 'error' key at top-level results
    if "error" in results:
        return True
    return False


def _get_game_key(data: Dict[str, Any]) -> Optional[str]:
    meta = data.get("metadata")
    if isinstance(meta, dict):
        game_name = meta.get("game")
        if isinstance(game_name, str) and game_name.strip():
            return game_name.strip()
    return None


def _evaluate_one(
    *,
    filepath: str,
    judge_name: str,
    providers_config_path: str,
    mode: Any,
    metrics: Optional[List[str]]
) -> bool:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read JSON: {filepath} ({e})")
        return False

    if "interaction_log" not in data:
        print(f"❌ Missing 'interaction_log' in: {os.path.basename(filepath)} — skipping")
        return False

    game_key = _get_game_key(data)
    if not game_key:
        print(f"❌ Missing metadata.game in: {os.path.basename(filepath)} — skipping")
        return False

    # Initialize judge
    try:
        judge_llm = load_llm_provider(providers_config_path, judge_name)
    except Exception as e:
        print(f"❌ Cannot load judge provider '{judge_name}': {e}")
        return False

    evaluator = HybridEvaluator(
        judge_llm_provider=judge_llm,
        game=game_key,
        mode=_normalize_mode(mode),
    )

    trial_data = {"interaction_log": data["interaction_log"]}

    try:
        results = evaluator.evaluate_trial(trial_data=trial_data, metrics=metrics)
    except Exception as e:
        print(f"❌ Evaluation error on {os.path.basename(filepath)}: {e}")
        return False

    new_eval: Dict[str, Any] = {"mode": mode, "results": results}
    data["evaluation"] = new_eval

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"❌ Failed writing JSON: {filepath} ({e})")
        return False

    print(f"✅ Re-evaluated: {os.path.basename(filepath)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run evaluation for result logs missing/errored evaluations")
    parser.add_argument(
        "--config",
        "-c",
        default=os.environ.get("RE_EVAL_CONFIG", os.path.join(_project_root(), "config", "re_evaluate.yaml")),
        help="Path to re-evaluate YAML config (default: ./config/re_evaluate.yaml or RE_EVAL_CONFIG env)",
    )
    parser.add_argument(
        "--providers-config",
        "-p",
        default=os.environ.get("PROVIDERS_CONFIG", os.path.join(_project_root(), "config", "config.yaml")),
        help="Path to providers config YAML (default: ./config/config.yaml or PROVIDERS_CONFIG env)",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)

    results_glob: str = cfg.get("results_glob", "results/**/*.json")
    mode: Any = cfg.get("mode", "both")
    judge_name: str = cfg.get("judge_llm_provider", "qwen3_14b_local_vllm_native")
    limit: int = int(cfg.get("limit", 0) or 0)
    metrics_cfg = cfg.get("metrics")
    metrics: Optional[List[str]] = list(metrics_cfg) if isinstance(metrics_cfg, list) else None

    # Concurrency settings
    concurrency: int = int(cfg.get("concurrency", 0) or 0)
    judge_pool_cfg = cfg.get("judge_pool")
    judge_pool: List[str] = [str(x) for x in judge_pool_cfg] if isinstance(judge_pool_cfg, list) else []
    if not judge_pool and concurrency > 0:
        judge_pool = [judge_name for _ in range(concurrency)]

    # Resolve glob path relative to project root if not absolute
    pattern = results_glob if os.path.isabs(results_glob) else os.path.join(_project_root(), results_glob)

    files = sorted(glob.glob(pattern, recursive=True))
    if limit > 0:
        files = files[:limit]

    if not files:
        print(f"No files matched: {results_glob}")
        return

    # Pre-scan to determine which files need re-evaluation
    candidates: List[str] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # unreadable files are skipped silently here; could be logged
            continue
        if _needs_reeval(data):
            candidates.append(fp)

    if not candidates:
        print("All matched files already contain valid evaluation. Nothing to do.")
        return

    print(f"Found {len(candidates)} files requiring re-evaluation.")

    reeval_count = 0

    if concurrency <= 0 or len(candidates) == 1:
        # Sequential path
        for fp in candidates:
            ok = _evaluate_one(
                filepath=fp,
                judge_name=judge_name,
                providers_config_path=args.providers_config,
                mode=mode,
                metrics=metrics,
            )
            if ok:
                reeval_count += 1
    else:
        # Parallel execution with provider rotation
        pool_cycle = cycle(judge_pool if judge_pool else [judge_name])
        max_workers = max(1, int(concurrency))
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for fp in candidates:
                futures.append(
                    executor.submit(
                        _evaluate_one,
                        filepath=fp,
                        judge_name=next(pool_cycle),
                        providers_config_path=args.providers_config,
                        mode=mode,
                        metrics=metrics,
                    )
                )

            for fut in as_completed(futures):
                try:
                    if fut.result():
                        reeval_count += 1
                except Exception:
                    # Any unexpected worker error is treated as a failed re-evaluation
                    pass

    print(f"\nDone. Re-evaluated: {reeval_count}, Skipped (already OK): {len(candidates) - reeval_count}")


if __name__ == "__main__":
    # Ensure src on path if run from project root
    sys.path.insert(0, _project_root())
    main()
