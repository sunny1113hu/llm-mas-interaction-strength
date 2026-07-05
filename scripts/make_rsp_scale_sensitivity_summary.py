from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


TASK_METRICS = {
    "consensus": "final_dispersion",
    "diffusion": "roughness_final",
}


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = float(statistics.mean(values))
    if len(values) < 2:
        return mean, 0.0
    ci95 = 1.96 * float(statistics.stdev(values)) / math.sqrt(len(values))
    return mean, ci95


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summarize_run(run_dir: Path, scale_factor: float, task: str) -> dict:
    metric = TASK_METRICS[task]
    rows = _load_rows(run_dir / "trial_summary.jsonl")
    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(float(row["mu"]), []).append(float(row[metric]))

    best_mu = min(grouped, key=lambda mu: statistics.mean(grouped[mu]))
    best_mean, best_ci95 = _mean_ci95(grouped[best_mu])
    low_mu = min(grouped)
    high_mu = max(grouped)
    low_mean, low_ci95 = _mean_ci95(grouped[low_mu])
    high_mean, high_ci95 = _mean_ci95(grouped[high_mu])
    return {
        "task": task,
        "scale_factor": f"{scale_factor:.2f}",
        "run_id": run_dir.name,
        "metric": metric,
        "n_trials": len(rows),
        "n_mu": len(grouped),
        "best_mu": f"{best_mu:.2f}",
        "best_mean": f"{best_mean:.3f}",
        "best_ci95": f"{best_ci95:.3f}",
        "low_mu": f"{low_mu:.2f}",
        "low_mu_mean": f"{low_mean:.3f}",
        "low_mu_ci95": f"{low_ci95:.3f}",
        "high_mu": f"{high_mu:.2f}",
        "high_mu_mean": f"{high_mean:.3f}",
        "high_mu_ci95": f"{high_ci95:.3f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RSP rule scale-sensitivity sweeps.")
    parser.add_argument("--outputs-root", default="outputs/runs")
    parser.add_argument("--out", default="outputs/runs/rsp_scale_sensitivity_summary.csv")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    specs = [
        (0.25, "s025"),
        (0.50, "s050"),
        (1.00, "s100"),
    ]
    rows = []
    for scale_factor, tag in specs:
        for task in TASK_METRICS:
            run_dir = outputs_root / f"rsp_scale_{tag}_{task}"
            rows.append(_summarize_run(run_dir, scale_factor, task))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
