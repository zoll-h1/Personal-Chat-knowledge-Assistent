from __future__ import annotations

from functools import lru_cache
from time import perf_counter

from fastapi import APIRouter

from app.core.config import Settings, get_settings
from app.rag.answer import AnswerGenerator
from app.rag.embeddings import LocalEmbedder
from app.rag.qdrant_store import QdrantStore
from app.rag.retriever import Retriever
from app.rag.schema import AskRequest, AskResponse, Citation

router = APIRouter()


class ChatService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = LocalEmbedder(
            model_name=settings.emb_model_name,
            batch_size=settings.emb_batch_size,
            normalize_embeddings=settings.emb_normalize,
        )
        self.store = QdrantStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
            vector_size=settings.emb_vector_size,
            timeout_s=settings.qdrant_timeout_s,
        )
        self.retriever = Retriever(self.embedder, self.store, settings)
        self.answerer = AnswerGenerator(settings)

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def ask(self, request: AskRequest) -> AskResponse:
        started = perf_counter()
        contexts = self.retriever.retrieve(
            question=request.question,
            top_k=request.top_k,
            topic=request.topic,
            date_from=request.date_from,
            date_to=request.date_to,
            chat_ids=request.chat_ids,
        )
        answer, confidence = self.answerer.generate(
            question=request.question,
            contexts=contexts,
            mode=request.mode,
        )

        citations = [
            Citation(
                chat_id=ctx.chat_id,
                message_ids=ctx.message_ids,
                snippet=(ctx.text[:300] + "...") if len(ctx.text) > 300 else ctx.text,
                score=ctx.score,
                created_at=ctx.created_at,
            )
            for ctx in contexts
        ]

        latency_ms = (perf_counter() - started) * 1000
        return AskResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            latency_ms=latency_ms,
        )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(settings)


@router.get("/health")
def health() -> dict[str, str]:
    return get_chat_service().health()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return get_chat_service().ask(request)
