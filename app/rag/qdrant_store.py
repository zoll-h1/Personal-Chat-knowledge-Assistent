from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.logging import get_logger
from app.rag.schema import ChunkRecord, RetrievalContext

logger = get_logger(__name__)


class QdrantStore:
    def __init__(self, url: str, collection_name: str, vector_size: int, timeout_s: float = 10.0) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(url=url, timeout=timeout_s)

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)

    def create_collection(self, reset: bool = False) -> None:
        exists = self.collection_exists()
        if exists and reset:
            logger.info("Deleting existing collection %s", self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info("Creating collection %s", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(size=self.vector_size, distance=qm.Distance.COSINE),
            )

        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        for field_name, schema_type in (
            ("topic", qm.PayloadSchemaType.KEYWORD),
            ("chat_id", qm.PayloadSchemaType.KEYWORD),
            ("created_at_ts", qm.PayloadSchemaType.FLOAT),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                # Index creation is idempotent in practice; tolerate incompatibilities.
                pass

    def _iso_to_ts(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None

    def _to_payload(self, chunk: ChunkRecord) -> dict[str, Any]:
        created_ts = self._iso_to_ts(chunk.start_at)
        return {
            "chunk_id": chunk.chunk_id,
            "chat_id": chunk.chat_id,
            "chat_title": chunk.chat_title,
            "topic": chunk.topic,
            "created_at_start": chunk.start_at,
            "created_at_end": chunk.end_at,
            "created_at_ts": created_ts,
            "message_ids": chunk.message_ids,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: np.ndarray, batch_size: int = 64) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunks count must match vectors count")
        if not chunks:
            return

        self.create_collection(reset=False)

        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            batch_chunks = chunks[start:end]
            batch_vectors = vectors[start:end]

            points = [
                qm.PointStruct(
                    id=chunk.chunk_id,
                    vector=batch_vectors[idx].tolist(),
                    payload=self._to_payload(chunk),
                )
                for idx, chunk in enumerate(batch_chunks)
            ]

            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def _build_filter(
        self,
        topic: str | None,
        date_from: str | None,
        date_to: str | None,
        chat_ids: list[str] | None,
    ) -> qm.Filter | None:
        must: list[qm.FieldCondition] = []

        if topic:
            must.append(
                qm.FieldCondition(
                    key="topic",
                    match=qm.MatchValue(value=topic),
                )
            )

        if chat_ids:
            must.append(
                qm.FieldCondition(
                    key="chat_id",
                    match=qm.MatchAny(any=chat_ids),
                )
            )

        from_ts = self._iso_to_ts(date_from)
        to_ts = self._iso_to_ts(date_to)
        if from_ts is not None or to_ts is not None:
            must.append(
                qm.FieldCondition(
                    key="created_at_ts",
                    range=qm.Range(gte=from_ts, lte=to_ts),
                )
            )

        if not must:
            return None
        return qm.Filter(must=must)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        topic: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        chat_ids: list[str] | None = None,
    ) -> list[RetrievalContext]:
        if not self.collection_exists():
            return []

        query_filter = self._build_filter(topic, date_from, date_to, chat_ids)
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        contexts: list[RetrievalContext] = []
        for hit in hits:
            payload = hit.payload or {}
            contexts.append(
                RetrievalContext(
                    chunk_id=str(payload.get("chunk_id") or hit.id),
                    chat_id=str(payload.get("chat_id", "unknown")),
                    chat_title=payload.get("chat_title"),
                    message_ids=[str(mid) for mid in payload.get("message_ids", [])],
                    topic=str(payload.get("topic", "unknown")),
                    text=str(payload.get("text", "")),
                    score=float(hit.score),
                    created_at=payload.get("created_at_start"),
                )
            )

        return contexts

    def stats(self) -> dict[str, Any]:
        if not self.collection_exists():
            return {
                "collection_name": self.collection_name,
                "points_count": 0,
                "indexed_vectors_count": 0,
            }

        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "points_count": int(info.points_count or 0),
            "indexed_vectors_count": int(info.indexed_vectors_count or 0),
        }

    def delete_by_chat_id(self, chat_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id))]
                )
            ),
            wait=True,
        )
