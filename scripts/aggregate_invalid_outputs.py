"""Aggregate the rate of invalid LLM outputs (fallback to the previous state).

An update is counted as invalid when the recorded raw output cannot be turned
into an integer state by the same coercion logic used at simulation time
(``src.swarm.policy.LLMPolicy.act``). In that case the simulator retains the
value from the previous time step.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.swarm.policy import _parse_json_output

PAPER_RUNS = [
    "20260109_102923_consensus_4x4_4llms",
    "20260115_073051_diffusion_4x4_4llms",
    "20260116_110026_consensus_ministral14B",
    "20260116_110026_diffusion_ministral14B",
    "20260117_174520_consensus_5seeds",
    "20260117_174520_diffusion_5seeds",
    "20260120_003407_consensus",
    "20260120_003407_diffusion",
    "20260124_010246_consensus_add5nopi",
    "20260124_010246_diffusion_add5nopi",
    "20260125_111339_consensus_T=30",
    "20260125_111339_diffusionT=30",
    "20260204_082317_consensus_6x6",
    "20260206_103341_consensus_8x8",
    "20260317_diffusion_6x6_diffusion",
    "20260317_diffusion_8x8_diffusion",
]


def _is_invalid(record: dict) -> bool:
    parsed, parse_ok = _parse_json_output(record.get("raw_output") or "")
    if not parse_ok:
        return True
    value = parsed.get("value")
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)
    if isinstance(value, float):
        value = int(round(value))
    return not isinstance(value, int)


def _is_llm_update(record: dict) -> bool:
    """Anchor cells are overwritten by the environment and never call the LLM."""
    if int(record.get("step", 0)) == 0:
        return False
    return str(record.get("role")) != "anchor"


def _task_of(record: dict, run_id: str) -> str:
    task = record.get("task_type")
    if task:
        return str(task)
    return "diffusion" if "diffusion" in run_id else "consensus"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/runs/invalid_output_rates.csv"))
    args = parser.parse_args()

    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    for run_id in PAPER_RUNS:
        path = args.outputs_root / run_id / "steps.jsonl"
        if not path.exists():
            print(f"[skip] {run_id}: steps.jsonl not found")
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not _is_llm_update(record):
                    continue
                key = (str(record.get("model_id")), _task_of(record, run_id))
                counts[key][0] += 1
                if _is_invalid(record):
                    counts[key][1] += 1
        print(f"[done] {run_id}")

    rows = []
    for (model, task), (total, invalid) in sorted(counts.items()):
        rate = 100.0 * invalid / total if total else 0.0
        rows.append((model, task, total, invalid, rate))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write("model,task,n_updates,n_invalid,invalid_rate_percent\n")
        for model, task, total, invalid, rate in rows:
            handle.write(f"{model},{task},{total},{invalid},{rate:.4f}\n")

    grand_total = sum(r[2] for r in rows)
    grand_invalid = sum(r[3] for r in rows)
    print()
    print(f"{'model':<32} {'task':<10} {'updates':>10} {'invalid':>8} {'rate %':>8}")
    for model, task, total, invalid, rate in rows:
        print(f"{model:<32} {task:<10} {total:>10} {invalid:>8} {rate:>8.3f}")
    overall = 100.0 * grand_invalid / grand_total if grand_total else 0.0
    print(f"\nOVERALL: {grand_invalid} / {grand_total} = {overall:.3f}%")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
