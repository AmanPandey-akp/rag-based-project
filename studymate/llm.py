"""Free, local LLM client (Ollama).

Ollama runs open-weight models (Llama, Qwen, Gemma, Phi ...) on your own machine.
It needs no API key, has no usage limits and costs nothing.
"""

from __future__ import annotations

import json
from typing import Iterator

import requests

from . import config


class LLMUnavailable(RuntimeError):
    """Raised when Ollama is not running or the model has not been downloaded."""


class OllamaClient:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or config.LLM_MODEL
        self.base_url = (base_url or config.OLLAMA_URL).rstrip("/")

    # ---------------------------------------------------------------- status
    def installed_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.base_url}. Install it from https://ollama.com "
                "and make sure it is running (open the Ollama app, or run `ollama serve`)."
            ) from exc
        return [m["name"] for m in r.json().get("models", [])]

    def has_model(self) -> bool:
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in self.installed_models()

    def check(self) -> None:
        """Raise LLMUnavailable with a helpful message if we cannot generate."""
        if not self.has_model():
            raise LLMUnavailable(
                f"Model '{self.model}' is not downloaded yet. Run:  ollama pull {self.model}"
            )

    # ------------------------------------------------------------ generation
    def stream(self, messages: list[dict], temperature: float | None = None) -> Iterator[str]:
        """Yield the answer token-by-token (so the UI/terminal feels instant)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
                "num_ctx": config.LLM_CONTEXT_TOKENS,
            },
        }
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=config.LLM_TIMEOUT_SECONDS,
            ) as r:
                if r.status_code == 404:
                    raise LLMUnavailable(
                        f"Model '{self.model}' not found. Run:  ollama pull {self.model}"
                    )
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise LLMUnavailable(data["error"])
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
        except requests.ConnectionError as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.base_url}. Is it running? (`ollama serve`)"
            ) from exc

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        return "".join(self.stream(messages, temperature))
