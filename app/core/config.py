from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Chat Assistant"
    log_level: str = "INFO"

    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "chat_chunks"
    qdrant_timeout_s: float = 10.0

    emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_vector_size: int = 384
    emb_batch_size: int = 32
    emb_normalize: bool = True

    max_chunk_tokens: int = 900
    overlap_messages: int = 2

    top_k_default: int = 10
    confidence_threshold: float = 0.35

    mode: Literal["extractive", "llm"] = "extractive"
    hybrid_keyword: bool = False
    enable_rerank: bool = False

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")

    allowlist_it_only: bool = False
    exclude_title_keywords: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 8501
    api_base_url: str = "http://api:8000"

    @property
    def messages_jsonl_path(self) -> Path:
        return self.processed_data_dir / "messages.jsonl"

    @property
    def chunks_jsonl_path(self) -> Path:
        return self.processed_data_dir / "chunks.jsonl"

    @property
    def exclude_title_keywords_list(self) -> list[str]:
        return [k.strip().lower() for k in self.exclude_title_keywords.split(",") if k.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
