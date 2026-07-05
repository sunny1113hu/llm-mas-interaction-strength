from __future__ import annotations

from dataclasses import dataclass

from src.llm_backend.base import LLMBackend, LLMConfig


@dataclass
class PythonAPIBackend(LLMBackend):
    config: LLMConfig

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            "Python API backend is not implemented yet. Use openai_compatible backend."
        )
