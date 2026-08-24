"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
RUNTIME_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embed_model: str = Field(default="text-embedding-3-small", alias="EMBED_MODEL")

    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_timeout_s: float = Field(default=60.0, alias="LLM_TIMEOUT_S")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")

    retriever_top_k: int = Field(default=4, alias="RETRIEVER_TOP_K")
    use_embeddings: bool = Field(default=True, alias="USE_EMBEDDINGS")

    session_ttl_s: int = Field(default=60 * 60 * 6, alias="SESSION_TTL_S")
    history_max_messages: int = Field(default=20, alias="HISTORY_MAX_MESSAGES")

    # A plain string rather than list[str]: pydantic-settings parses complex types
    # from .env as JSON and blows up on "a,b".
    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173", alias="CORS_ORIGINS"
    )

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]

    @property
    def base_url(self) -> str:
        return self.openai_base_url.rstrip("/")

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
