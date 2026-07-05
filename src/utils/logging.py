from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.data import Problem


@dataclass
class RunLogger:
    run_dir: Path
    run_id: str
    trial_id: int
    seed: int
    problem_id: str
    task_type: str | None = None
    model_id: str | None = None
    mu: float | None = None
    sigma: float | None = None
    prompt_mode: str | None = None
    policy_kind: str | None = None

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.run_dir / "steps.jsonl"
        self.trials_path = self.run_dir / "trial_summary.jsonl"

    def log_step(
        self,
        step: int,
        agent_id: int,
        output_json: Dict[str, Any],
        local_context: str,
        raw_output: str,
        parse_ok: bool,
        neighbors_higher: int,
        neighbors_lower: int,
        neighbors_equal: int,
        prev_value: int | None,
        action: str | None = None,
        accepted: bool | None = None,
        mode: str | None = None,
        min_value: int | None = None,
        max_value: int | None = None,
        mu: float | None = None,
        sigma: float | None = None,
        epsilon: float | None = None,
        tau_i: float | None = None,
        p_i: float | None = None,
        role: str | None = None,
    ) -> None:
        value = output_json.get("value")
        delta = None
        if isinstance(value, int) and isinstance(prev_value, int):
            delta = value - prev_value
        record = {
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "seed": self.seed,
            "task_type": self.task_type,
            "model_id": self.model_id,
            "mu": mu if mu is not None else self.mu,
            "sigma": sigma if sigma is not None else self.sigma,
            "prompt_mode": self.prompt_mode,
            "policy_kind": self.policy_kind,
            "step": step,
            "agent_id": agent_id,
            "value": value,
            "prev_value": prev_value,
            "delta": delta,
            "neighbors_higher": neighbors_higher,
            "neighbors_lower": neighbors_lower,
            "neighbors_equal": neighbors_equal,
            "action": action,
            "accepted": accepted,
            "mode": mode,
            "min_value": min_value,
            "max_value": max_value,
            "epsilon": epsilon,
            "tau_i": tau_i,
            "p_i": p_i,
            "role": role,
            "local_context": local_context,
            "rationale": output_json.get("rationale", ""),
            "raw_output": raw_output,
            "parse_ok": parse_ok,
        }
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def summarize_trial(
        self,
        problem: Problem,
        final_outputs: List[Dict[str, Any]],
        duration_s: float,
        steps: int,
        grid_size: int,
        initial_values: List[int],
        extra: Dict[str, Any] | None = None,
        min_value: int | None = None,
        max_value: int | None = None,
        mu: float | None = None,
        sigma: float | None = None,
    ) -> Dict[str, Any]:
        final_values = [out.get("value") for out in final_outputs if isinstance(out.get("value"), int)]
        consensus_final = bool(final_values) and len(set(final_values)) == 1
        final_consensus_value = final_values[0] if consensus_final else None
        final_dispersion = None
        if final_values:
            mean_val = sum(final_values) / len(final_values)
            final_dispersion = sum(abs(v - mean_val) for v in final_values) / len(final_values)
        initial_dispersion = None
        if initial_values:
            mean_init = sum(initial_values) / len(initial_values)
            initial_dispersion = sum(abs(v - mean_init) for v in initial_values) / len(initial_values)
        summary = {
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "problem_id": problem.problem_id,
            "seed": self.seed,
            "task_type": self.task_type,
            "model_id": self.model_id,
            "mu": mu if mu is not None else self.mu,
            "sigma": sigma if sigma is not None else self.sigma,
            "prompt_mode": self.prompt_mode,
            "policy_kind": self.policy_kind,
            "steps": steps,
            "grid_size": grid_size,
            "min_value": min_value,
            "max_value": max_value,
            "final_consensus_value": final_consensus_value,
            "consensus_final": consensus_final,
            "final_dispersion": final_dispersion,
            "initial_dispersion": initial_dispersion,
            "duration_s": duration_s,
        }
        if extra:
            summary.update(extra)
        with self.trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=True) + "\n")
        return summary
