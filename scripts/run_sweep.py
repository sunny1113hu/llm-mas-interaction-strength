from __future__ import annotations

import argparse
import copy
import json
import sys
import os
import time
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

import httpx

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


def _wait_for_vllm(base_url: str, timeout_s: int = 600, interval_s: float = 2.0) -> None:
    url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                if response.status_code == 200:
                    return
        except Exception:
            pass
        time.sleep(interval_s)
    raise RuntimeError(f"vLLM did not become ready at {url} within {timeout_s}s.")


def _restart_vllm(model_id: str, model_subdir: str, max_model_len: int | None, gpu_memory_utilization: float | None) -> None:
    if not shutil.which("docker"):
        raise RuntimeError("docker is required for multi-model sweep. Run this script on the host.")
    try:
        subprocess.run(["docker", "compose", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        raise RuntimeError("docker compose is required for multi-model sweep. Run this script on the host.") from exc
    env = os.environ.copy()
    env["MODEL_ID"] = model_id
    env["MODEL_SUBDIR"] = model_subdir
    if max_model_len is not None:
        env["MAX_MODEL_LEN"] = str(int(max_model_len))
    if gpu_memory_utilization is not None:
        env["GPU_MEMORY_UTILIZATION"] = str(float(gpu_memory_utilization))
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build", "--force-recreate", "vllm"],
        check=True,
        env=env,
    )


def _run_sweep_for_task(
    config: dict,
    run_id: str,
    task_type: str,
    model_list: list[dict],
    multi_model: bool,
    stop_vllm: bool,
    outputs_root: Path,
) -> None:
    run_dir = outputs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(config, run_dir / "config.yaml")
    run_start = time.perf_counter()

    _write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "mode": "run_sweep",
            "task_type": task_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_list": model_list,
        },
    )

    sim_cfg = config.get("simulation", {})
    mu_list = sim_cfg.get("mu_list")
    if mu_list is None:
        raise RuntimeError("mu_list is missing in config.")
    include_pi = bool(config.get("prompt", {}).get("include_pi", True))
    if not include_pi:
        mu_list = [float(mu_list[0])] if isinstance(mu_list, list) and mu_list else [0.0]
    trial_id = 0
    dataset_cfg = config.get("dataset", {})
    if task_type == "consensus":
        task_cfg = dataset_cfg.get("consensus", {})
    elif task_type == "diffusion":
        task_cfg = dataset_cfg.get("diffusion", {})
    else:
        raise RuntimeError(f"Unsupported dataset.type: {task_type}")
    raw_sigma_list = task_cfg.get("sigma_list")
    if isinstance(raw_sigma_list, list) and raw_sigma_list:
        sigma_list = [float(x) for x in raw_sigma_list]
    else:
        sigma_list = [float(task_cfg.get("sigma", 1.0))]
    seed_list = config["simulation"].get("seed_list") or [config["simulation"]["seed"]]
    raw_range_list = task_cfg.get("range_list")
    if isinstance(raw_range_list, list) and raw_range_list:
        range_list = raw_range_list
    else:
        range_list = [None]
    for model_entry in model_list:
        model_id = str(model_entry.get("model_id", "")).strip()
        if not model_id:
            raise RuntimeError("model_id is missing in models.model_list.")
        policy_kind = str(model_entry.get("policy_kind", config.get("policy", {}).get("kind", "llm")))
        model_subdir = model_entry.get("model_subdir")
        max_model_len = model_entry.get("max_model_len")
        gpu_memory_utilization = model_entry.get("gpu_memory_utilization")
        if multi_model and model_subdir and policy_requires_backend(policy_kind):
            _restart_vllm(
                model_id=model_id,
                model_subdir=str(model_subdir),
                max_model_len=int(max_model_len) if max_model_len is not None else None,
                gpu_memory_utilization=float(gpu_memory_utilization) if gpu_memory_utilization is not None else None,
            )
            _wait_for_vllm(config["llm"]["base_url"])

        config["llm"]["model"] = model_id
        config.setdefault("policy", {})
        config["policy"]["kind"] = policy_kind
        backend = build_backend(config["llm"]) if policy_requires_backend(policy_kind) else None

        prompt_mode = "with_pi" if include_pi else "no_pi"

        for seed in seed_list:
            config["simulation"]["seed"] = int(seed)
            for bounds in range_list:
                if bounds is not None and len(bounds) == 2:
                    task_cfg["min_value"] = int(bounds[0])
                    task_cfg["max_value"] = int(bounds[1])
                if task_type == "consensus":
                    config["dataset"]["consensus"] = task_cfg
                else:
                    config["dataset"]["diffusion"] = task_cfg
                problems = _load_problems(config=config, max_problems=config["run_sweep"]["num_problems"])
                for sigma in sigma_list:
                    task_cfg["sigma"] = float(sigma)
                    if task_type == "consensus":
                        config["dataset"]["consensus"] = task_cfg
                    else:
                        config["dataset"]["diffusion"] = task_cfg
                    for mu in mu_list:
                        for problem in problems:
                            logger = RunLogger(
                                run_dir=run_dir,
                                run_id=run_id,
                                trial_id=trial_id,
                                seed=config["simulation"]["seed"],
                                problem_id=problem.problem_id,
                                model_id=model_id,
                                task_type=task_type,
                                mu=float(mu),
                                sigma=float(task_cfg.get("sigma", 1.0)),
                                prompt_mode=prompt_mode,
                                policy_kind=policy_kind,
                            )
                            run_trial(
                                problem=problem,
                                backend=backend,
                                config=config,
                                mu=mu,
                                logger=logger,
                            )
                            trial_id += 1

    run_end = time.perf_counter()
    _write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "mode": "run_sweep",
            "task_type": task_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "duration_s": run_end - run_start,
            "model_list": model_list,
        },
    )
    if config.get("run_sweep", {}).get("make_figures", False):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.make_figures",
                "--run-id",
                run_id,
                "--outputs-root",
                str(outputs_root),
            ],
            check=False,
        )
    if stop_vllm and multi_model:
        try:
            subprocess.run(
                ["docker", "compose", "stop", "vllm"],
                check=False,
            )
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep mu values for phase-transition analysis.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stop-vllm", action="store_true")
    parser.add_argument("--mobility-scale-factor", type=float, default=None)
    parser.add_argument("--no-make-figures", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mobility_scale_factor is not None:
        config.setdefault("policy", {})
        config["policy"]["mobility_scale_factor"] = float(args.mobility_scale_factor)
    if args.no_make_figures:
        config.setdefault("run_sweep", {})
        config["run_sweep"]["make_figures"] = False
    sweep_cfg = config["run_sweep"]
    if os.getenv("LLM_BASE_URL"):
        config["llm"]["base_url"] = os.environ["LLM_BASE_URL"]

    outputs_root = Path(config["paths"]["outputs_root"])

    multi_model = bool(sweep_cfg.get("multi_model", False))
    if multi_model:
        model_list = config.get("models", {}).get("model_list") or []
        if not model_list:
            raise RuntimeError("models.model_list is required when run_sweep.multi_model is true.")
    else:
        model_list = [
            {
                "model_id": str(config.get("llm", {}).get("model", "")),
                "policy_kind": str(config.get("policy", {}).get("kind", "llm")),
            }
        ]

    dataset_cfg = config.get("dataset", {})
    task_list = sweep_cfg.get("task_list")
    if isinstance(task_list, list) and task_list:
        base_run_id = args.run_id or default_run_id()
        for task_type in task_list:
            task_type = str(task_type)
            task_config = copy.deepcopy(config)
            task_config["dataset"]["type"] = task_type
            run_id = f"{base_run_id}_{task_type}"
            _run_sweep_for_task(
                config=task_config,
                run_id=run_id,
                task_type=task_type,
                model_list=model_list,
                multi_model=multi_model,
                stop_vllm=False,
                outputs_root=outputs_root,
            )
        if args.stop_vllm and multi_model:
            try:
                subprocess.run(
                    ["docker", "compose", "stop", "vllm"],
                    check=False,
                )
            except Exception:
                pass
        return

    task_type = str(dataset_cfg.get("type", "consensus"))
    run_id = args.run_id or default_run_id()
    _run_sweep_for_task(
        config=config,
        run_id=run_id,
        task_type=task_type,
        model_list=model_list,
        multi_model=multi_model,
        stop_vllm=args.stop_vllm,
        outputs_root=outputs_root,
    )


if __name__ == "__main__":
    main()
