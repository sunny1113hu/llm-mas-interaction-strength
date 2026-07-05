from __future__ import annotations

import random
from typing import Dict, List

from src.data.types import Problem


def load_consensus(
    max_problems: int,
    config: Dict[str, object],
    seed: int,
) -> List[Problem]:
    min_value = config.get("min_value")
    max_value = config.get("max_value")
    if min_value is None or max_value is None:
        range_list = config.get("range_list")
        if isinstance(range_list, list) and range_list:
            bounds = range_list[0]
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                min_value = bounds[0]
                max_value = bounds[1]
    min_value = int(min_value) if min_value is not None else 0
    max_value = int(max_value) if max_value is not None else 50
    grid_size = int(config.get("grid_size", 4))
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")
    num_agents = grid_size ** 2
    rng = random.Random(seed)

    problems: List[Problem] = []
    for idx in range(max_problems):
        initial_values = [rng.randint(min_value, max_value) for _ in range(num_agents)]
        local_contexts = {
            agent_id: f"initial_value: {value}"
            for agent_id, value in enumerate(initial_values)
        }
        problems.append(
            Problem(
                problem_id=f"consensus-{idx}",
                context="",
                question="Reach consensus on a single integer with your neighbors.",
                answer="",
                evidence_ids=[],
                local_contexts=local_contexts,
                metadata={
                    "min_value": min_value,
                    "max_value": max_value,
                    "initial_values": initial_values,
                },
            )
        )
    return problems
