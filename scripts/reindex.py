from __future__ import annotations

import argparse
from pathlib import Path

from app.api.routes_chat import get_chat_service
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.chunking import load_chunks_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex existing chunks into Qdrant")
    parser.add_argument("--chunks", default=None, help="Optional custom chunks JSONL path")
    parser.add_argument("--reset", action="store_true", help="Reset collection before indexing")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    service = get_chat_service()

    chunks_path = Path(args.chunks) if args.chunks else settings.chunks_jsonl_path
    chunks = load_chunks_jsonl(chunks_path)

    service.store.create_collection(reset=args.reset)
    if not chunks:
        print(f"No chunks found at {chunks_path}")
        return

    vectors = service.embedder.embed_texts([chunk.text for chunk in chunks])
    service.store.upsert_chunks(chunks, vectors)
    print({"indexed_chunks": len(chunks), "chunks_path": str(chunks_path)})


if __name__ == "__main__":
    main()
