from __future__ import annotations

import argparse
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config_snapshot
from src.data import load_consensus, load_diffusion
from src.llm_backend import build_backend
from src.swarm.policy import policy_requires_backend
from src.swarm.simulator import default_run_id, run_trial
from src.utils.logging import RunLogger


def _write_manifest(run_dir: Path, payload: dict) -> None:
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _load_problems(config: dict, max_problems: int) -> list:
    dataset_cfg = config["dataset"]
    task_type = dataset_cfg.get("type", "consensus")
    if task_type == "consensus":
        consensus_cfg = dict(dataset_cfg.get("consensus", {}))
        consensus_cfg["grid_size"] = config["simulation"]["grid_size"]
        return load_consensus(
            max_problems=max_problems,
            config=consensus_cfg,
            seed=config["simulation"]["seed"],
        )
    if task_type == "diffusion":
        task_cfg = dict(dataset_cfg.get("diffusion", {}))
        task_cfg["grid_size"] = config["simulation"]["grid_size"]
        return load_diffusion(
            max_problems=max_problems,
            config=task_cfg,
            seed=config["simulation"]["seed"],
        )
    raise RuntimeError(f"Unsupported dataset.type: {task_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal E2E swarm trial.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_cfg = config["run_one"]
    if os.getenv("LLM_BASE_URL"):
        config["llm"]["base_url"] = os.environ["LLM_BASE_URL"]

    sim_cfg = dict(config["simulation"])
    sim_cfg["steps"] = run_cfg["steps"]
    config["simulation"] = sim_cfg

    run_id = args.run_id or default_run_id()
    outputs_root = Path(config["paths"]["outputs_root"])
    run_dir = outputs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(config, run_dir / "config.yaml")
    run_start = time.perf_counter()
    _write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "mode": "run_one",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )

    problems = _load_problems(config=config, max_problems=run_cfg["num_problems"])

    policy_kind = str(config.get("policy", {}).get("kind", "llm"))
    backend = build_backend(config["llm"]) if policy_requires_backend(policy_kind) else None
    dataset_cfg = config.get("dataset", {})
    task_type = str(dataset_cfg.get("type", "consensus"))
    if task_type == "consensus":
        task_cfg = dataset_cfg.get("consensus", {})
    elif task_type == "diffusion":
        task_cfg = dataset_cfg.get("diffusion", {})
    else:
        raise RuntimeError(f"Unsupported dataset.type: {task_type}")
    model_id = str(config.get("llm", {}).get("model", ""))
    include_pi = bool(config.get("prompt", {}).get("include_pi", True))
    prompt_mode = "with_pi" if include_pi else "no_pi"

    mu_value = float(run_cfg.get("mu", 0.0))
    for trial_id, problem in enumerate(problems):
        logger = RunLogger(
            run_dir=run_dir,
            run_id=run_id,
            trial_id=trial_id,
            seed=sim_cfg["seed"],
            problem_id=problem.problem_id,
            model_id=model_id,
            task_type=task_type,
            mu=mu_value,
            sigma=float(task_cfg.get("sigma", 0.33)),
            prompt_mode=prompt_mode,
            policy_kind=policy_kind,
        )
        run_trial(
            problem=problem,
            backend=backend,
            config=config,
            mu=mu_value,
            logger=logger,
        )
    run_end = time.perf_counter()
    _write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "mode": "run_one",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "duration_s": run_end - run_start,
        },
    )


if __name__ == "__main__":
    main()
