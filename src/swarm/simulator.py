from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.data import Problem
from src.llm_backend.base import LLMBackend
from src.swarm.grid import Grid
from src.swarm.policy import Observation, build_policy
from src.utils.logging import RunLogger


@dataclass
class TrialResult:
    summary: Dict[str, Any]


def _stable_seed(seed: int, key: str) -> int:
    import hashlib

    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _neighbors_open(agent_id: int, size: int) -> list[int]:
    r, c = divmod(agent_id, size)
    neighbors: list[int] = []
    if r > 0:
        neighbors.append((r - 1) * size + c)
    if r < size - 1:
        neighbors.append((r + 1) * size + c)
    if c > 0:
        neighbors.append(r * size + (c - 1))
    if c < size - 1:
        neighbors.append(r * size + (c + 1))
    return neighbors


def _ideal_gradient(size: int, source_value: float, sink_value: float) -> np.ndarray:
    grid = np.full((size, size), (source_value + sink_value) / 2.0, dtype=float)
    grid[0, 0] = source_value
    grid[size - 1, size - 1] = sink_value
    max_iter = 5000
    tol = 1e-4
    for _ in range(max_iter):
        max_delta = 0.0
        for r in range(size):
            for c in range(size):
                if (r == 0 and c == 0) or (r == size - 1 and c == size - 1):
                    continue
                neighbors = []
                if r > 0:
                    neighbors.append(grid[r - 1, c])
                if r < size - 1:
                    neighbors.append(grid[r + 1, c])
                if c > 0:
                    neighbors.append(grid[r, c - 1])
                if c < size - 1:
                    neighbors.append(grid[r, c + 1])
                avg = sum(neighbors) / len(neighbors)
                delta = abs(avg - grid[r, c])
                if delta > max_delta:
                    max_delta = delta
                grid[r, c] = avg
        if max_delta < tol:
            break
    return grid


def _roughness(values: list[int], size: int) -> float:
    total = 0.0
    edges = 0
    for r in range(size):
        for c in range(size):
            idx = r * size + c
            if r < size - 1:
                down = (r + 1) * size + c
                total += (values[idx] - values[down]) ** 2
                edges += 1
            if c < size - 1:
                right = r * size + (c + 1)
                total += (values[idx] - values[right]) ** 2
                edges += 1
    return total / edges if edges else 0.0


def default_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def run_trial(
    problem: Problem,
    backend: Optional[LLMBackend],
    config: Dict[str, Any],
    mu: float,
    logger: RunLogger,
) -> TrialResult:
    task_type = str(config.get("dataset", {}).get("type", "consensus"))
    if task_type == "diffusion":
        return _run_diffusion(problem, backend, config, mu, logger)
    return _run_consensus(problem, backend, config, mu, logger)


def _run_consensus(
    problem: Problem,
    backend: Optional[LLMBackend],
    config: Dict[str, Any],
    mu: float,
    logger: RunLogger,
) -> TrialResult:
    sim_cfg = config["simulation"]
    grid = Grid(size=sim_cfg["grid_size"])
    consensus_cfg = config.get("dataset", {}).get("consensus", {})

    sigma = float(consensus_cfg.get("sigma", 1.0))
    mu_logit = float(mu)

    if problem.metadata:
        min_value = int(problem.metadata.get("min_value", consensus_cfg.get("min_value", 0)))
        max_value = int(problem.metadata.get("max_value", consensus_cfg.get("max_value", 50)))
        initial_values = list(problem.metadata.get("initial_values") or [])
    else:
        min_value = int(consensus_cfg.get("min_value", 0))
        max_value = int(consensus_cfg.get("max_value", 50))
        initial_values = []

    if not initial_values:
        initial_values = [0 for _ in range(grid.size ** 2)]

    eps_rng = random.Random(_stable_seed(sim_cfg["seed"] + 271, problem.problem_id))
    consensus_params: List[Dict[str, Any]] = []
    for _ in range(grid.size ** 2):
        epsilon = eps_rng.gauss(0.0, 1.0)
        tau_i = float(mu_logit) + float(sigma) * epsilon
        p_i = _sigmoid(tau_i)
        if p_i < 0.3:
            role = "stubborn"
        elif p_i > 0.7:
            role = "unstable"
        else:
            role = "sensible"
        consensus_params.append(
            {
                "epsilon": epsilon,
                "tau_i": tau_i,
                "p_i": p_i,
                "role": role,
            }
        )

    current_values = list(initial_values)
    last_outputs: List[Dict[str, Any]] = []
    policy_kind = str(config.get("policy", {}).get("kind", "llm"))
    policy = build_policy(policy_kind=policy_kind, backend=backend)

    start_time = time.perf_counter()
    initial_outputs: List[Dict[str, Any]] = []
    for agent_id in range(grid.size ** 2):
        neighbor_ids = grid.neighbors(agent_id)
        prev_value = current_values[agent_id]
        neighbor_vals = [current_values[n_id] for n_id in neighbor_ids]
        neighbors_higher = sum(1 for v in neighbor_vals if v > prev_value)
        neighbors_lower = sum(1 for v in neighbor_vals if v < prev_value)
        neighbors_equal = sum(1 for v in neighbor_vals if v == prev_value)
        local_context = f"initial_value: {prev_value}"
        init_output = {"value": prev_value, "rationale": "Initial state."}
        initial_outputs.append(init_output)
        logger.log_step(
            step=0,
            agent_id=agent_id,
            output_json=init_output,
            local_context=local_context,
            raw_output="",
            parse_ok=True,
            neighbors_higher=neighbors_higher,
            neighbors_lower=neighbors_lower,
            neighbors_equal=neighbors_equal,
            prev_value=prev_value,
            min_value=min_value,
            max_value=max_value,
            mu=mu_logit,
            sigma=sigma,
            epsilon=consensus_params[agent_id]["epsilon"],
            tau_i=consensus_params[agent_id]["tau_i"],
            p_i=consensus_params[agent_id]["p_i"],
            role=consensus_params[agent_id]["role"],
        )
    last_outputs = initial_outputs

    include_pi = bool(config.get("prompt", {}).get("include_pi", True))
    mobility_scale_factor = float(config.get("policy", {}).get("mobility_scale_factor", 0.5))
    for step in range(1, sim_cfg["steps"] + 1):
        step_outputs: List[Dict[str, Any]] = []
        for agent_id in range(grid.size ** 2):
            neighbor_ids = grid.neighbors(agent_id)
            neighbor_vals = [current_values[n_id] for n_id in neighbor_ids]
            prev_value = current_values[agent_id]
            neighbors_higher = sum(1 for v in neighbor_vals if v > prev_value)
            neighbors_lower = sum(1 for v in neighbor_vals if v < prev_value)
            neighbors_equal = sum(1 for v in neighbor_vals if v == prev_value)
            decision = policy.act(
                Observation(
                    task_type="consensus",
                    current_value=prev_value,
                    neighbors_higher=neighbors_higher,
                    neighbors_lower=neighbors_lower,
                    neighbors_equal=neighbors_equal,
                    min_value=min_value,
                    max_value=max_value,
                    grid_size=grid.size,
                    include_pi=include_pi,
                    p_i=consensus_params[agent_id]["p_i"],
                    mobility_scale_factor=mobility_scale_factor,
                )
            )
            output = {
                "value": decision.value,
                "rationale": decision.rationale,
                "role": consensus_params[agent_id]["role"],
            }
            step_outputs.append(output)
            logger.log_step(
                step=step,
                agent_id=agent_id,
                output_json=output,
                local_context=f"initial_value: {prev_value}",
                raw_output=decision.raw_output,
                parse_ok=decision.parse_ok,
                neighbors_higher=neighbors_higher,
                neighbors_lower=neighbors_lower,
                neighbors_equal=neighbors_equal,
                prev_value=prev_value,
                min_value=min_value,
                max_value=max_value,
                mu=mu_logit,
                sigma=sigma,
                epsilon=consensus_params[agent_id]["epsilon"],
                tau_i=consensus_params[agent_id]["tau_i"],
                p_i=consensus_params[agent_id]["p_i"],
                role=consensus_params[agent_id]["role"],
            )
        last_outputs = step_outputs
        current_values = [out.get("value", current_values[idx]) for idx, out in enumerate(step_outputs)]

    duration_s = time.perf_counter() - start_time
    summary = logger.summarize_trial(
        problem=problem,
        final_outputs=last_outputs,
        duration_s=duration_s,
        steps=sim_cfg["steps"],
        grid_size=grid.size,
        initial_values=initial_values,
        extra={"mobility_scale_factor": mobility_scale_factor},
        min_value=min_value,
        max_value=max_value,
        mu=mu_logit,
        sigma=sigma,
    )
    return TrialResult(summary=summary)



def _run_diffusion(
    problem: Problem,
    backend: Optional[LLMBackend],
    config: Dict[str, Any],
    mu: float,
    logger: RunLogger,
) -> TrialResult:
    sim_cfg = config["simulation"]
    grid = Grid(size=sim_cfg["grid_size"])
    task_cfg = config.get("dataset", {}).get("diffusion", {})

    sigma = float(task_cfg.get("sigma", 0.5))
    mu_logit = float(mu)

    if problem.metadata:
        min_value = int(problem.metadata.get("min_value", task_cfg.get("min_value", 0)))
        max_value = int(problem.metadata.get("max_value", task_cfg.get("max_value", 25)))
        initial_values = list(problem.metadata.get("initial_values") or [])
        source_id = int(problem.metadata.get("source_id", 0))
        sink_id = int(problem.metadata.get("sink_id", grid.size * grid.size - 1))
        source_value = int(problem.metadata.get("source_value", max_value))
        sink_value = int(problem.metadata.get("sink_value", min_value))
    else:
        min_value = int(task_cfg.get("min_value", 0))
        max_value = int(task_cfg.get("max_value", 25))
        initial_values = []
        source_id = 0
        sink_id = grid.size * grid.size - 1
        source_value = int(task_cfg.get("source_value", max_value))
        sink_value = int(task_cfg.get("sink_value", min_value))

    if not initial_values:
        rng = random.Random(_stable_seed(sim_cfg["seed"], problem.problem_id))
        initial_values = [rng.randint(min_value, max_value) for _ in range(grid.size ** 2)]
    initial_values[source_id] = source_value
    initial_values[sink_id] = sink_value

    eps_rng = random.Random(_stable_seed(sim_cfg["seed"] + 271, problem.problem_id))
    agent_params: List[Dict[str, Any]] = []
    for _ in range(grid.size ** 2):
        epsilon = eps_rng.gauss(0.0, 1.0)
        tau_i = float(mu_logit) + float(sigma) * epsilon
        p_i = _sigmoid(tau_i)
        agent_params.append({"epsilon": epsilon, "tau_i": tau_i, "p_i": p_i})

    current_values = list(initial_values)
    last_outputs: List[Dict[str, Any]] = []
    ideal = _ideal_gradient(grid.size, float(source_value), float(sink_value))
    policy_kind = str(config.get("policy", {}).get("kind", "llm"))
    policy = build_policy(policy_kind=policy_kind, backend=backend)

    def _mse(values: list[int]) -> float:
        total = 0.0
        for idx, val in enumerate(values):
            r, c = divmod(idx, grid.size)
            total += (float(val) - ideal[r, c]) ** 2
        return total / len(values)

    mse_series: List[float] = []
    rough_series: List[float] = []

    start_time = time.perf_counter()
    for agent_id in range(grid.size ** 2):
        prev_value = current_values[agent_id]
        neighbor_ids = _neighbors_open(agent_id, grid.size)
        neighbor_vals = [current_values[n_id] for n_id in neighbor_ids]
        neighbors_higher = sum(1 for v in neighbor_vals if v > prev_value)
        neighbors_lower = sum(1 for v in neighbor_vals if v < prev_value)
        neighbors_equal = sum(1 for v in neighbor_vals if v == prev_value)
        init_output = {"value": prev_value, "rationale": "Initial state."}
        last_outputs.append(init_output)
        logger.log_step(
            step=0,
            agent_id=agent_id,
            output_json=init_output,
            local_context=f"initial_value: {prev_value}",
            raw_output="",
            parse_ok=True,
            neighbors_higher=neighbors_higher,
            neighbors_lower=neighbors_lower,
            neighbors_equal=neighbors_equal,
            prev_value=prev_value,
            min_value=min_value,
            max_value=max_value,
            mu=mu_logit,
            sigma=sigma,
            epsilon=agent_params[agent_id]["epsilon"],
            tau_i=agent_params[agent_id]["tau_i"],
            p_i=agent_params[agent_id]["p_i"],
            role="anchor" if agent_id in (source_id, sink_id) else "free",
            action="anchor" if agent_id in (source_id, sink_id) else "init",
            accepted=False,
            mode="init",
        )

    mse_series.append(_mse(current_values))
    rough_series.append(_roughness(current_values, grid.size))

    include_pi = bool(config.get("prompt", {}).get("include_pi", True))
    mobility_scale_factor = float(config.get("policy", {}).get("mobility_scale_factor", 0.5))
    for step in range(1, sim_cfg["steps"] + 1):
        next_values = list(current_values)
        for agent_id in range(grid.size ** 2):
            if agent_id in (source_id, sink_id):
                next_values[agent_id] = source_value if agent_id == source_id else sink_value
                prev_value = current_values[agent_id]
                neighbor_ids = _neighbors_open(agent_id, grid.size)
                neighbor_vals = [current_values[n_id] for n_id in neighbor_ids]
                neighbors_higher = sum(1 for v in neighbor_vals if v > prev_value)
                neighbors_lower = sum(1 for v in neighbor_vals if v < prev_value)
                neighbors_equal = sum(1 for v in neighbor_vals if v == prev_value)
                output = {"value": next_values[agent_id], "rationale": "Anchor fixed."}
                logger.log_step(
                    step=step,
                    agent_id=agent_id,
                    output_json=output,
                    local_context=f"anchor_value: {prev_value}",
                    raw_output="",
                    parse_ok=True,
                    neighbors_higher=neighbors_higher,
                    neighbors_lower=neighbors_lower,
                    neighbors_equal=neighbors_equal,
                    prev_value=prev_value,
                    min_value=min_value,
                    max_value=max_value,
                    mu=mu_logit,
                    sigma=sigma,
                    epsilon=agent_params[agent_id]["epsilon"],
                    tau_i=agent_params[agent_id]["tau_i"],
                    p_i=agent_params[agent_id]["p_i"],
                    role="anchor",
                    action="anchor",
                    accepted=False,
                    mode="diffusion",
                )
                continue
            neighbor_ids = _neighbors_open(agent_id, grid.size)
            neighbor_vals = [current_values[n_id] for n_id in neighbor_ids]
            prev_value = current_values[agent_id]
            neighbors_higher = sum(1 for v in neighbor_vals if v > prev_value)
            neighbors_lower = sum(1 for v in neighbor_vals if v < prev_value)
            neighbors_equal = sum(1 for v in neighbor_vals if v == prev_value)
            decision = policy.act(
                Observation(
                    task_type="diffusion",
                    current_value=prev_value,
                    neighbors_higher=neighbors_higher,
                    neighbors_lower=neighbors_lower,
                    neighbors_equal=neighbors_equal,
                    min_value=min_value,
                    max_value=max_value,
                    grid_size=grid.size,
                    include_pi=include_pi,
                    p_i=agent_params[agent_id]["p_i"],
                    mobility_scale_factor=mobility_scale_factor,
                )
            )
            next_values[agent_id] = decision.value
            output = {"value": decision.value, "rationale": decision.rationale}
            logger.log_step(
                step=step,
                agent_id=agent_id,
                output_json=output,
                local_context=f"current_value: {prev_value}",
                raw_output=decision.raw_output,
                parse_ok=decision.parse_ok,
                neighbors_higher=neighbors_higher,
                neighbors_lower=neighbors_lower,
                neighbors_equal=neighbors_equal,
                prev_value=prev_value,
                min_value=min_value,
                max_value=max_value,
                mu=mu_logit,
                sigma=sigma,
                epsilon=agent_params[agent_id]["epsilon"],
                tau_i=agent_params[agent_id]["tau_i"],
                p_i=agent_params[agent_id]["p_i"],
                role="free",
                action="update",
                accepted=True,
                mode="diffusion",
            )
        current_values = list(next_values)
        mse_series.append(_mse(current_values))
        rough_series.append(_roughness(current_values, grid.size))

    duration_s = time.perf_counter() - start_time
    mse_initial = mse_series[0] if mse_series else 0.0
    mse_final = mse_series[-1] if mse_series else 0.0
    rough_initial = rough_series[0] if rough_series else 0.0
    rough_final = rough_series[-1] if rough_series else 0.0
    mse_reduction = (mse_initial - mse_final) / max(mse_initial, 1e-9)

    summary = logger.summarize_trial(
        problem=problem,
        final_outputs=[{"value": v} for v in current_values],
        duration_s=duration_s,
        steps=sim_cfg["steps"],
        grid_size=grid.size,
        initial_values=initial_values,
        min_value=min_value,
        max_value=max_value,
        mu=mu_logit,
        sigma=sigma,
        extra={
            "mse_initial": mse_initial,
            "mse_final": mse_final,
            "roughness_initial": rough_initial,
            "roughness_final": rough_final,
            "mse_reduction": mse_reduction,
            "source_id": source_id,
            "sink_id": sink_id,
            "source_value": source_value,
            "sink_value": sink_value,
            "mobility_scale_factor": mobility_scale_factor,
        },
    )
    return TrialResult(summary=summary)
