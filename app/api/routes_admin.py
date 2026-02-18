from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.routes_chat import get_chat_service
from app.core.config import get_settings
from app.rag.chunking import build_chunks, load_chunks_jsonl, write_chunks_jsonl
from app.rag.ingest.export_reader import ingest_export, load_messages_jsonl, resolve_input_path
from app.rag.schema import AdminStatsResponse, IngestRequest, ReindexRequest

router = APIRouter(prefix="/admin", tags=["admin"])


def _service():
    return get_chat_service()


def _settings():
    return get_settings()


@router.post("/ingest")
def ingest_endpoint(request: IngestRequest) -> dict[str, Any]:
    settings = _settings()
    service = _service()

    try:
        input_path = resolve_input_path(settings.raw_data_dir, request.input_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    allowlist_it_only = (
        request.allowlist_it_only if request.allowlist_it_only is not None else settings.allowlist_it_only
    )
    exclude_title_keywords = (
        request.exclude_title_keywords
        if request.exclude_title_keywords is not None
        else settings.exclude_title_keywords_list
    )

    summary = ingest_export(
        input_path=input_path,
        output_messages_path=settings.messages_jsonl_path,
        allowlist_it_only=allowlist_it_only,
        exclude_title_keywords=exclude_title_keywords,
    )

    messages = load_messages_jsonl(settings.messages_jsonl_path)
    chunks = build_chunks(
        messages,
        max_tokens=settings.max_chunk_tokens,
        overlap_messages=settings.overlap_messages,
    )
    write_chunks_jsonl(settings.chunks_jsonl_path, chunks)

    service.store.create_collection(reset=False)
    if chunks:
        vectors = service.embedder.embed_texts([chunk.text for chunk in chunks])
        service.store.upsert_chunks(chunks, vectors)

    summary["chunk_count"] = len(chunks)
    summary["output_chunks_path"] = str(settings.chunks_jsonl_path)
    return summary


@router.post("/reindex")
def reindex_endpoint(request: ReindexRequest) -> dict[str, Any]:
    settings = _settings()
    service = _service()

    chunks_path = Path(request.chunks_path) if request.chunks_path else settings.chunks_jsonl_path
    chunks = load_chunks_jsonl(chunks_path)

    service.store.create_collection(reset=request.reset_collection)
    if not chunks:
        return {
            "collection_name": settings.collection_name,
            "indexed_chunks": 0,
            "message": f"No chunks found at {chunks_path}",
        }

    vectors = service.embedder.embed_texts([chunk.text for chunk in chunks])
    service.store.upsert_chunks(chunks, vectors)

    return {
        "collection_name": settings.collection_name,
        "indexed_chunks": len(chunks),
        "chunks_path": str(chunks_path),
    }


@router.get("/stats", response_model=AdminStatsResponse)
def stats_endpoint() -> AdminStatsResponse:
    settings = _settings()
    service = _service()

    qdrant_stats = service.store.stats()
    messages_count = len(load_messages_jsonl(settings.messages_jsonl_path))
    chunks_count = len(load_chunks_jsonl(settings.chunks_jsonl_path))

    return AdminStatsResponse(
        collection_name=qdrant_stats["collection_name"],
        points_count=qdrant_stats["points_count"],
        indexed_vectors_count=qdrant_stats["indexed_vectors_count"],
        messages_count=messages_count,
        chunks_count=chunks_count,
    )


@router.post("/collection/reset")
def reset_collection_endpoint() -> dict[str, str]:
    settings = _settings()
    service = _service()
    service.store.create_collection(reset=True)
    return {"status": "ok", "collection_name": settings.collection_name}


@router.delete("/chats/{chat_id}")
def delete_chat_endpoint(chat_id: str) -> dict[str, str]:
    service = _service()
    service.store.delete_by_chat_id(chat_id)
    return {"status": "ok", "chat_id": chat_id}
