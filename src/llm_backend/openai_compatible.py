from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from src.llm_backend.base import LLMBackend, LLMConfig


@dataclass
class OpenAICompatibleBackend(LLMBackend):
    config: LLMConfig

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        try:
            with httpx.Client(timeout=self.config.timeout_s) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Failed to reach vLLM server. Verify base_url/model and that the server is running."
            ) from exc
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned from vLLM: {json.dumps(data)[:200]}")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"Empty response content from vLLM: {json.dumps(data)[:200]}")
        return content
