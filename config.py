"""RAID-SQL configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPIDER = (
    PROJECT_ROOT.parent / "Few-shot-NL2SQL-with-prompting" / "data"
)


class Settings(BaseSettings):
    """Runtime settings loaded from environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chat_provider: str = Field(default="openai", alias="CHAT_PROVIDER")

    # Gemini / Vertex
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    google_cloud_project: Optional[str] = Field(
        default=None, alias="GOOGLE_CLOUD_PROJECT"
    )
    google_cloud_location: str = Field(
        default="us-central1", alias="GOOGLE_CLOUD_LOCATION"
    )
    google_application_credentials: Optional[str] = Field(
        default=None, alias="GOOGLE_APPLICATION_CREDENTIALS"
    )

    # OpenAI chat
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")

    llm_max_retries: int = Field(default=8, alias="LLM_MAX_RETRIES")
    llm_retry_base_sec: float = Field(default=2.0, alias="LLM_RETRY_BASE_SEC")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    max_output_tokens: int = Field(default=600, alias="MAX_OUTPUT_TOKENS")

    # Published list prices (USD per 1M tokens) — update when vendors change.
    gemini_flash_input_per_m: float = Field(
        default=0.30, alias="GEMINI_FLASH_INPUT_PER_M"
    )
    gemini_flash_output_per_m: float = Field(
        default=2.50, alias="GEMINI_FLASH_OUTPUT_PER_M"
    )
    gemini_pro_input_per_m: float = Field(
        default=1.25, alias="GEMINI_PRO_INPUT_PER_M"
    )
    gemini_pro_output_per_m: float = Field(
        default=10.00, alias="GEMINI_PRO_OUTPUT_PER_M"
    )
    openai_input_per_m: float = Field(default=2.50, alias="OPENAI_INPUT_PER_M")
    openai_output_per_m: float = Field(default=10.00, alias="OPENAI_OUTPUT_PER_M")
    openai_embedding_per_m: float = Field(
        default=0.02, alias="OPENAI_EMBEDDING_PER_M"
    )
    # Vertex text-embedding-004 ≈ $0.025 / 1M chars; token estimate ≈ chars/4
    gemini_embedding_per_m: float = Field(
        default=0.10, alias="GEMINI_EMBEDDING_PER_M"
    )
    gemini_embedding_dims: Optional[int] = Field(
        default=768, alias="GEMINI_EMBEDDING_DIMS"
    )

    # DIN-SQL paper comparators (GPT-4)
    din_sql_paper_cost_per_query_usd: float = Field(
        default=0.50, alias="DIN_SQL_PAPER_COST_PER_QUERY_USD"
    )
    din_sql_paper_latency_sec: float = Field(
        default=60.0, alias="DIN_SQL_PAPER_LATENCY_SEC"
    )

    max_budget_usd: Optional[float] = Field(default=None, alias="MAX_BUDGET_USD")

    # Retrieval / embeddings
    embedding_provider: str = Field(default="openai", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    chroma_path: str = Field(default=".chroma", alias="CHROMA_PATH")
    fewshot_collection: str = Field(
        default="spider_train_fewshot_openai", alias="FEWSHOT_COLLECTION"
    )
    fewshot_top_k: int = Field(default=6, alias="FEWSHOT_TOP_K")
    use_rag: bool = Field(default=True, alias="USE_RAG")
    include_db_values: bool = Field(default=True, alias="INCLUDE_DB_VALUES")
    db_value_limit: int = Field(default=5, alias="DB_VALUE_LIMIT")
    # RAID-SQL v2: RAG-first demos. Set true to restore legacy DIN static banks.
    use_din_static_banks: bool = Field(default=False, alias="USE_DIN_STATIC_BANKS")

    # Pipeline
    max_repair_attempts: int = Field(default=3, alias="MAX_REPAIR_ATTEMPTS")
    repair_on_empty: bool = Field(default=True, alias="REPAIR_ON_EMPTY")
    self_consistency_k: int = Field(default=5, alias="SELF_CONSISTENCY_K")
    self_consistency_temperature: float = Field(
        default=0.6, alias="SELF_CONSISTENCY_TEMPERATURE"
    )
    use_execution_repair: bool = Field(default=True, alias="USE_EXECUTION_REPAIR")
    use_self_consistency: bool = Field(default=True, alias="USE_SELF_CONSISTENCY")
    use_text_debug: bool = Field(default=True, alias="USE_TEXT_DEBUG")
    debug_max_output_tokens: int = Field(default=600, alias="DEBUG_MAX_OUTPUT_TOKENS")

    spider_data_dir: str = Field(
        default=str(DEFAULT_SPIDER), alias="SPIDER_DATA_DIR"
    )

    @property
    def chroma_dir(self) -> Path:
        path = Path(self.chroma_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def spider_dir(self) -> Path:
        return Path(self.spider_data_dir).expanduser().resolve()

    @property
    def chat_model(self) -> str:
        provider = (self.chat_provider or "openai").lower()
        if provider == "openai":
            return self.openai_model
        return self.gemini_model

    def rates_for_model(self, model: str) -> tuple[float, float]:
        """Return (input_per_m, output_per_m) USD rates."""
        name = (model or "").lower()
        provider = (self.chat_provider or "").lower()
        if provider == "openai" or name.startswith("gpt-") or "gpt-4" in name:
            return self.openai_input_per_m, self.openai_output_per_m
        if "pro" in name:
            return self.gemini_pro_input_per_m, self.gemini_pro_output_per_m
        return self.gemini_flash_input_per_m, self.gemini_flash_output_per_m


def get_settings() -> Settings:
    return Settings()
