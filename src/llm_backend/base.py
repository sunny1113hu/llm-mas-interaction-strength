from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class LLMConfig:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_s: int
    response_format: Optional[str] = None


class LLMBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...
