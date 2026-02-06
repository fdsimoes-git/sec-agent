import ollama

from .base import ModelProvider


class OllamaProvider(ModelProvider):
    """Ollama-backed LLM provider."""

    def __init__(self, model: str = "qwen2.5-coder:7b"):
        self.model = model

    def chat(self, messages: list[dict]) -> str:
        response = ollama.chat(
            model=self.model,
            messages=messages,
        )
        return response["message"]["content"]
