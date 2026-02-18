from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Qdrant collection")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collection")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    store = QdrantStore(
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
        vector_size=settings.emb_vector_size,
        timeout_s=settings.qdrant_timeout_s,
    )
    store.create_collection(reset=args.reset)
    print(f"Collection ready: {settings.collection_name}")


if __name__ == "__main__":
    main()
