import ollama

from .base import ModelProvider


class OllamaProvider(ModelProvider):
    """Ollama-backed LLM provider."""

    def __init__(self, model: str = "qwen2.5-coder:7b", num_ctx: int = 6000):
        self.model = model
        self.num_ctx = num_ctx

    def chat(self, messages: list[dict]) -> str:
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"num_ctx": self.num_ctx},
        )
        return response["message"]["content"]
