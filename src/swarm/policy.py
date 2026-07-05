from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Protocol

from src.llm_backend.base import LLMBackend
from src.swarm.agent import build_diffusion_prompt, build_prompt


@dataclass
class Observation:
    task_type: str
    current_value: int
    neighbors_higher: int
    neighbors_lower: int
    neighbors_equal: int
    min_value: int
    max_value: int
    grid_size: int
    include_pi: bool
    p_i: Optional[float] = None
    mobility_scale_factor: float = 0.5


@dataclass
class PolicyDecision:
    value: int
    rationale: str
    raw_output: str = ""
    parse_ok: bool = True


class DecisionPolicy(Protocol):
    def act(self, observation: Observation) -> PolicyDecision:
        ...


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def _normal_round(value: float) -> int:
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _mobility_scale(observation: Observation) -> float:
    return float(observation.mobility_scale_factor) * float(observation.max_value - observation.min_value)


def _parse_json_output(raw_text: str) -> tuple[dict, bool]:
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}")
        payload = raw_text[start : end + 1]
        data = json.loads(payload)
    except Exception:
        return {"value": None, "rationale": "Failed to parse model output."}, False
    value = data.get("value")
    rationale = str(data.get("rationale") or "").strip()
    return {"value": value, "rationale": rationale or "No rationale provided."}, True


class LLMPolicy:
    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.system_prompt = (
            "You are a precise coordination agent. Output only JSON and nothing else. "
            "Follow the user instructions strictly."
        )

    def act(self, observation: Observation) -> PolicyDecision:
        if observation.task_type == "diffusion":
            prompt = build_diffusion_prompt(
                current_value=observation.current_value,
                neighbors_higher=observation.neighbors_higher,
                neighbors_lower=observation.neighbors_lower,
                neighbors_equal=observation.neighbors_equal,
                p_i=float(observation.p_i or 0.0),
                min_value=observation.min_value,
                max_value=observation.max_value,
                grid_size=observation.grid_size,
                include_pi=observation.include_pi,
            )
        else:
            prompt = build_prompt(
                current_value=observation.current_value,
                neighbors_higher=observation.neighbors_higher,
                neighbors_lower=observation.neighbors_lower,
                neighbors_equal=observation.neighbors_equal,
                min_value=observation.min_value,
                max_value=observation.max_value,
                p_i=float(observation.p_i or 0.0),
                grid_size=observation.grid_size,
                include_pi=observation.include_pi,
            )
        raw_output = self.backend.generate(self.system_prompt, prompt)
        parsed_output, parse_ok = _parse_json_output(raw_output)
        value = parsed_output.get("value")
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        if isinstance(value, float):
            value = int(round(value))
        if not isinstance(value, int):
            value = observation.current_value
        value = _clamp(value, observation.min_value, observation.max_value)
        return PolicyDecision(
            value=value,
            rationale=str(parsed_output.get("rationale", "")),
            raw_output=raw_output,
            parse_ok=parse_ok,
        )


class PressureScaledPolicy:
    def act(self, observation: Observation) -> PolicyDecision:
        if observation.p_i is None:
            raise RuntimeError("PressureScaledPolicy requires p_i in the observation.")
        delta = observation.neighbors_higher - observation.neighbors_lower
        step = _normal_round(float(observation.p_i) * delta)
        value = _clamp(observation.current_value + step, observation.min_value, observation.max_value)
        rationale = f"Pressure-scaled rule: delta={delta}, p_i={float(observation.p_i):.3f}, step={step}."
        return PolicyDecision(value=value, rationale=rationale)


class PressureRawPolicy:
    def act(self, observation: Observation) -> PolicyDecision:
        delta = observation.neighbors_higher - observation.neighbors_lower
        value = _clamp(observation.current_value + delta, observation.min_value, observation.max_value)
        rationale = f"Raw-pressure rule: delta={delta}, step={delta}."
        return PolicyDecision(value=value, rationale=rationale)


class MajorityUnitPolicy:
    def act(self, observation: Observation) -> PolicyDecision:
        delta = observation.neighbors_higher - observation.neighbors_lower
        if delta > 0:
            step = 1
        elif delta < 0:
            step = -1
        else:
            step = 0
        value = _clamp(observation.current_value + step, observation.min_value, observation.max_value)
        rationale = f"Majority-unit rule: delta={delta}, step={step}."
        return PolicyDecision(value=value, rationale=rationale)


class MobilityScaledPolicy:
    def act(self, observation: Observation) -> PolicyDecision:
        if observation.p_i is None:
            raise RuntimeError("MobilityScaledPolicy requires p_i in the observation.")
        delta = observation.neighbors_higher - observation.neighbors_lower
        normalized_pressure = delta / 4.0
        mobility_scale = _mobility_scale(observation)
        step = _normal_round(float(observation.p_i) * normalized_pressure * mobility_scale)
        value = _clamp(observation.current_value + step, observation.min_value, observation.max_value)
        rationale = (
            "Mobility-scaled rule: "
            f"delta={delta}, pressure={normalized_pressure:.3f}, "
            f"p_i={float(observation.p_i):.3f}, scale={mobility_scale:.3f}, step={step}."
        )
        return PolicyDecision(value=value, rationale=rationale)


class MobilityRawPolicy:
    def act(self, observation: Observation) -> PolicyDecision:
        delta = observation.neighbors_higher - observation.neighbors_lower
        normalized_pressure = delta / 4.0
        mobility_scale = _mobility_scale(observation)
        step = _normal_round(normalized_pressure * mobility_scale)
        value = _clamp(observation.current_value + step, observation.min_value, observation.max_value)
        rationale = (
            "Mobility-raw rule: "
            f"delta={delta}, pressure={normalized_pressure:.3f}, scale={mobility_scale:.3f}, step={step}."
        )
        return PolicyDecision(value=value, rationale=rationale)


def policy_requires_backend(policy_kind: str) -> bool:
    return policy_kind == "llm"


def build_policy(policy_kind: str, backend: Optional[LLMBackend]) -> DecisionPolicy:
    if policy_kind == "llm":
        if backend is None:
            raise RuntimeError("LLM policy requires a backend.")
        return LLMPolicy(backend)
    if policy_kind == "pressure_scaled":
        return PressureScaledPolicy()
    if policy_kind == "pressure_raw":
        return PressureRawPolicy()
    if policy_kind == "majority_unit":
        return MajorityUnitPolicy()
    if policy_kind == "mobility_scaled":
        return MobilityScaledPolicy()
    if policy_kind == "mobility_raw":
        return MobilityRawPolicy()
    raise ValueError(f"Unknown policy kind: {policy_kind}")
