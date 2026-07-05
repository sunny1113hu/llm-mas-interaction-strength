from __future__ import annotations

import random
from typing import Dict, List

from src.data.types import Problem


def load_diffusion(
    max_problems: int,
    config: Dict[str, object],
    seed: int,
) -> List[Problem]:
    grid_size = int(config.get("grid_size", 4))
    num_agents = grid_size**2
    rng = random.Random(seed)

    min_value = int(config.get("min_value", 0))
    max_value = int(config.get("max_value", 25))
    source_value = int(config.get("source_value", max_value))
    sink_value = int(config.get("sink_value", min_value))

    source_id = 0
    sink_id = num_agents - 1

    problems: List[Problem] = []
    for idx in range(max_problems):
        initial_values = [rng.randint(min_value, max_value) for _ in range(num_agents)]
        initial_values[source_id] = source_value
        initial_values[sink_id] = sink_value
        local_contexts = {
            agent_id: f"initial_value: {value}"
            for agent_id, value in enumerate(initial_values)
        }
        problems.append(
            Problem(
                problem_id=f"diffusion-{idx}",
                context="",
                question="Diffuse values to form a smooth gradient between fixed anchors.",
                answer="",
                evidence_ids=[],
                local_contexts=local_contexts,
                metadata={
                    "min_value": min_value,
                    "max_value": max_value,
                    "initial_values": initial_values,
                    "source_id": source_id,
                    "sink_id": sink_id,
                    "source_value": source_value,
                    "sink_value": sink_value,
                },
            )
        )
    return problems
