from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.rag.chunking import load_chunks_jsonl
from app.rag.qdrant_store import QdrantStore
from app.rag.reranker import LexicalReranker
from app.rag.schema import RetrievalContext

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{3,}")


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


class Retriever:
    def __init__(self, embedder, store: QdrantStore, settings: Settings) -> None:
        self.embedder = embedder
        self.store = store
        self.settings = settings
        self._reranker = LexicalReranker()

    def _to_timestamp(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None

    def _keyword_search(
        self,
        question: str,
        top_k: int,
        topic: str | None,
        date_from: str | None,
        date_to: str | None,
        chat_ids: list[str] | None,
    ) -> list[RetrievalContext]:
        chunks_path = Path(self.settings.chunks_jsonl_path)
        chunks = load_chunks_jsonl(chunks_path)
        if not chunks:
            return []

        query_terms = _tokens(question)
        if not query_terms:
            return []

        from_ts = self._to_timestamp(date_from)
        to_ts = self._to_timestamp(date_to)

        scored: list[tuple[float, RetrievalContext]] = []
        for chunk in chunks:
            if topic and chunk.topic != topic:
                continue
            if chat_ids and chunk.chat_id not in chat_ids:
                continue

            chunk_ts = self._to_timestamp(chunk.start_at)
            if from_ts is not None and chunk_ts is not None and chunk_ts < from_ts:
                continue
            if to_ts is not None and chunk_ts is not None and chunk_ts > to_ts:
                continue

            doc_terms = _tokens(chunk.text)
            if not doc_terms:
                continue
            overlap = len(query_terms & doc_terms)
            if overlap == 0:
                continue
            score = overlap / max(len(query_terms), 1)
            scored.append(
                (
                    score,
                    RetrievalContext(
                        chunk_id=chunk.chunk_id,
                        chat_id=chunk.chat_id,
                        chat_title=chunk.chat_title,
                        message_ids=chunk.message_ids,
                        topic=chunk.topic,
                        text=chunk.text,
                        score=float(score),
                        created_at=chunk.start_at,
                    ),
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [ctx for _, ctx in scored[:top_k]]

    def _merge_results(
        self,
        vector_results: list[RetrievalContext],
        keyword_results: list[RetrievalContext],
        top_k: int,
    ) -> list[RetrievalContext]:
        merged: dict[str, RetrievalContext] = {}
        for ctx in vector_results + keyword_results:
            existing = merged.get(ctx.chunk_id)
            if existing is None or ctx.score > existing.score:
                merged[ctx.chunk_id] = ctx

        results = list(merged.values())
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top_k]

    def retrieve(
        self,
        question: str,
        top_k: int,
        topic: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        chat_ids: list[str] | None = None,
    ) -> list[RetrievalContext]:
        query_vector = self.embedder.embed_query(question)
        vector_results = self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            topic=topic,
            date_from=date_from,
            date_to=date_to,
            chat_ids=chat_ids,
        )

        merged = vector_results
        if self.settings.hybrid_keyword:
            keyword_results = self._keyword_search(
                question=question,
                top_k=top_k,
                topic=topic,
                date_from=date_from,
                date_to=date_to,
                chat_ids=chat_ids,
            )
            merged = self._merge_results(vector_results, keyword_results, top_k=top_k * 2)

        if self.settings.enable_rerank:
            merged = self._reranker.rerank(question, merged, top_k=top_k)
        else:
            merged = merged[:top_k]

        return merged
