"""Recompute the receptivity-ablation table with held-out selection of mu.

The oracle variant reports the lowest seed-averaged metric over the swept mu,
which uses the evaluation seeds to pick mu. The leave-one-seed-out (LOO)
variant instead picks mu on the remaining seeds and evaluates on the held-out
seed, so no evaluation seed influences the choice of mu.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.analysis import load_jsonl

MODELS = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "microsoft/Phi-4-mini-instruct",
]
LABELS = {
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Ministral-3-8B-Instruct-2512": "Ministral-3-8B",
    "mistralai/Ministral-3-14B-Instruct-2512": "Ministral-3-14B",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
}

TASKS = {
    "consensus": {
        "metric": "final_dispersion",
        "with_pi": [
            "20260109_102923_consensus_4x4_4llms",
            "20260116_110026_consensus_ministral14B",
            "20260120_003407_consensus",
        ],
        "no_pi": [
            "20260117_174520_consensus_5seeds",
            "20260124_010246_consensus_add5nopi",
        ],
    },
    "diffusion": {
        "metric": "roughness_final",
        "with_pi": [
            "20260115_073051_diffusion_4x4_4llms",
            "20260116_110026_diffusion_ministral14B",
            "20260120_003407_diffusion",
        ],
        "no_pi": [
            "20260117_174520_diffusion_5seeds",
            "20260124_010246_diffusion_add5nopi",
        ],
    },
}


def _load(outputs_root: Path, run_ids: list[str]) -> list[dict]:
    trials: list[dict] = []
    for run_id in run_ids:
        trials.extend(load_jsonl(outputs_root / run_id / "trial_summary.jsonl"))
    return trials


def _filter(trials: list[dict], model_id: str, prompt_mode: str) -> list[dict]:
    out = []
    for record in trials:
        if record.get("model_id") != model_id:
            continue
        mode = record.get("prompt_mode") or "with_pi"
        if prompt_mode == "with_pi" and mode == "no_pi":
            continue
        if prompt_mode == "no_pi" and mode != "no_pi":
            continue
        if int(record.get("grid_size", 4)) != 4:
            continue
        if int(record.get("steps", 10)) != 10:
            continue
        out.append(record)
    return out


def _table(trials: list[dict], metric: str) -> dict[float, dict[int, float]]:
    """metric value indexed by [mu][seed]."""
    grid: dict[float, dict[int, float]] = defaultdict(dict)
    for record in trials:
        mu = record.get("mu")
        seed = record.get("seed")
        value = record.get(metric)
        if mu is None or seed is None or not isinstance(value, (int, float)):
            continue
        grid[float(mu)][int(seed)] = float(value)
    return dict(grid)


def _oracle(grid: dict[float, dict[int, float]]) -> tuple[float, float]:
    """Lowest seed-averaged metric over mu; returns (value, mu)."""
    best_mu, best_val = None, float("inf")
    for mu, by_seed in grid.items():
        mean = float(np.mean(list(by_seed.values())))
        if mean < best_val:
            best_mu, best_val = mu, mean
    return best_val, float(best_mu)


def _loo(grid: dict[float, dict[int, float]]) -> tuple[float, list[float], list[float]]:
    """Leave-one-seed-out selection of mu; returns (mean score, per-seed scores, chosen mus)."""
    seeds = sorted({s for by_seed in grid.values() for s in by_seed})
    scores: list[float] = []
    chosen: list[float] = []
    for held_out in seeds:
        best_mu, best_val = None, float("inf")
        for mu, by_seed in grid.items():
            others = [v for s, v in by_seed.items() if s != held_out]
            if not others:
                continue
            mean = float(np.mean(others))
            if mean < best_val:
                best_mu, best_val = mu, mean
        if best_mu is None or held_out not in grid[best_mu]:
            continue
        scores.append(grid[best_mu][held_out])
        chosen.append(best_mu)
    return float(np.mean(scores)), scores, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/runs/pi_ablation_summary.csv"))
    args = parser.parse_args()

    rows: list[dict] = []

    for task, spec in TASKS.items():
        metric = spec["metric"]
        with_pi_all = _load(args.outputs_root, spec["with_pi"])
        no_pi_all = _load(args.outputs_root, spec["no_pi"])
        print(f"\n===== {task} ({metric}) =====")
        header = f"{'model':<16} {'w/o Pi':>9} {'oracle':>9} {'(mu)':>7} {'red%':>7} {'LOO':>9} {'red%':>7} {'chosen mu':>22}"
        print(header)
        for model in MODELS:
            grid = _table(_filter(with_pi_all, model, "with_pi"), metric)
            no_pi = [
                float(r[metric])
                for r in _filter(no_pi_all, model, "no_pi")
                if isinstance(r.get(metric), (int, float))
            ]
            if not grid or not no_pi:
                print(f"{LABELS[model]:<16} (missing data)")
                continue
            base = float(np.mean(no_pi))
            oracle_val, oracle_mu = _oracle(grid)
            loo_val, _, chosen = _loo(grid)
            oracle_red = 100.0 * (base - oracle_val) / base
            loo_red = 100.0 * (base - loo_val) / base
            uniq = sorted(set(chosen))
            chosen_txt = ", ".join(f"{m:g}" for m in uniq)
            n_seeds = len({s for by in grid.values() for s in by})
            print(
                f"{LABELS[model]:<16} {base:>9.3g} {oracle_val:>9.3g} {oracle_mu:>7g}"
                f" {oracle_red:>+7.1f} {loo_val:>9.3g} {loo_red:>+7.1f} {chosen_txt:>22}"
                f"   (n_seeds={n_seeds}, n_nopi={len(no_pi)})"
            )
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "model": LABELS[model],
                    "n_seeds": n_seeds,
                    "no_pi_mean": base,
                    "oracle_mean": oracle_val,
                    "oracle_mu": oracle_mu,
                    "oracle_reduction_percent": oracle_red,
                    "heldout_loo_mean": loo_val,
                    "heldout_loo_reduction_percent": loo_red,
                    "loo_selected_mu": chosen_txt,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
