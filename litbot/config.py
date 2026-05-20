from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LITBOT_", extra="ignore")

    database_url: str = "postgresql://litbot:litbot@localhost:5432/litbot"
    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    vector_collection_name: str = "litbot_chunks"
    top_k: int = 8
    retrieval_candidate_multiplier: int = 8
    retrieval_min_candidates: int = 50
    retrieval_max_candidates: int = 200
    retrieval_rrf_k: int = 60
    retrieval_include_neighbors: bool = False
    retrieval_neighbor_window: int = 1
    prompt_version: str = "litbot-grounded-v1"
    note_prompt_version: str = "litbot-note-v1"
    intent_confidence_threshold: float = 0.65
    request_timeout_seconds: float = 60.0
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()
