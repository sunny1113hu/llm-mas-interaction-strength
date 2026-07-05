from src.llm_backend.base import LLMConfig
from src.llm_backend.openai_compatible import OpenAICompatibleBackend
from src.llm_backend.python_api import PythonAPIBackend


def build_backend(config_dict: dict):
    config = LLMConfig(
        base_url=config_dict["base_url"],
        model=config_dict["model"],
        temperature=config_dict["temperature"],
        max_tokens=config_dict["max_tokens"],
        timeout_s=config_dict["timeout_s"],
        response_format=config_dict.get("response_format"),
    )
    backend_name = config_dict.get("backend", "openai_compatible")
    if backend_name == "openai_compatible":
        return OpenAICompatibleBackend(config)
    if backend_name == "python_api":
        return PythonAPIBackend(config)
    raise ValueError(f"Unknown backend: {backend_name}")
