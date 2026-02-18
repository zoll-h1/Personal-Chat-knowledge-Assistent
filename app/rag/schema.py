from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Role = Literal["user", "assistant", "system", "tool", "unknown"]


class NormalizedMessage(BaseModel):
    chat_id: str
    chat_title: str | None = None
    message_id: str
    parent_message_id: str | None = None
    role: Role = "unknown"
    created_at: str | None = None
    text: str
    has_code: bool = False
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    topic: str = "unknown"
    source: Literal["chatgpt_export_json", "chatgpt_export_html"]


class ChunkRecord(BaseModel):
    chunk_id: str
    chat_id: str
    chat_title: str | None = None
    message_ids: list[str]
    start_at: str | None = None
    end_at: str | None = None
    topic: str = "unknown"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalContext(BaseModel):
    chunk_id: str
    chat_id: str
    chat_title: str | None = None
    message_ids: list[str]
    topic: str
    text: str
    score: float
    created_at: str | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=10, ge=1, le=50)
    topic: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    chat_ids: list[str] | None = None
    mode: Literal["extractive", "llm"] | None = None


class Citation(BaseModel):
    chat_id: str
    message_ids: list[str]
    snippet: str
    score: float
    created_at: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    latency_ms: float


class IngestRequest(BaseModel):
    input_path: str | None = None
    allowlist_it_only: bool | None = None
    exclude_title_keywords: list[str] | None = None


class ReindexRequest(BaseModel):
    reset_collection: bool = False
    chunks_path: str | None = None


class AdminStatsResponse(BaseModel):
    collection_name: str
    points_count: int
    indexed_vectors_count: int
    messages_count: int
    chunks_count: int
