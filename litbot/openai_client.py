import json
from typing import Any

from openai import OpenAI

from litbot.config import Settings, get_settings


class OpenAIModelClient:
    """Thin wrapper around OpenAI GPT and embedding APIs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        kwargs: dict[str, Any] = {"timeout": self.settings.request_timeout_seconds}
        if self.settings.openai_api_key:
            kwargs["api_key"] = self.settings.openai_api_key
        self.client = OpenAI(**kwargs)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
            dimensions=self.settings.embedding_dimensions,
        )
        return [item.embedding for item in response.data]

    def generate_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
