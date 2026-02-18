from __future__ import annotations

import argparse

from app.api.routes_chat import get_chat_service
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.chunking import build_chunks, write_chunks_jsonl
from app.rag.ingest.export_reader import ingest_export, load_messages_jsonl, resolve_input_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ChatGPT export into processed files and Qdrant")
    parser.add_argument("--input", dest="input_path", default=None, help="Path to ZIP/JSON/HTML export")
    parser.add_argument("--allowlist-it-only", action="store_true", help="Index only IT-related chats")
    parser.add_argument(
        "--exclude-title-keywords",
        default="",
        help="Comma-separated title keywords to exclude",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    service = get_chat_service()

    input_path = resolve_input_path(settings.raw_data_dir, args.input_path)
    exclude_keywords = [k.strip() for k in args.exclude_title_keywords.split(",") if k.strip()]

    summary = ingest_export(
        input_path=input_path,
        output_messages_path=settings.messages_jsonl_path,
        allowlist_it_only=args.allowlist_it_only or settings.allowlist_it_only,
        exclude_title_keywords=exclude_keywords or settings.exclude_title_keywords_list,
    )

    messages = load_messages_jsonl(settings.messages_jsonl_path)
    chunks = build_chunks(messages, max_tokens=settings.max_chunk_tokens, overlap_messages=settings.overlap_messages)
    write_chunks_jsonl(settings.chunks_jsonl_path, chunks)

    service.store.create_collection(reset=False)
    if chunks:
        vectors = service.embedder.embed_texts([chunk.text for chunk in chunks])
        service.store.upsert_chunks(chunks, vectors)

    print("Ingestion complete")
    print(summary)
    print({"chunk_count": len(chunks), "chunks_path": str(settings.chunks_jsonl_path)})


if __name__ == "__main__":
    main()
