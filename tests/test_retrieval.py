from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.rag.retriever import Retriever
from app.rag.schema import RetrievalContext


class FakeEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


class FakeStore:
    def __init__(self) -> None:
        self.last_kwargs = {}

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        return [
            RetrievalContext(
                chunk_id="c1",
                chat_id="chat-1",
                chat_title="title",
                message_ids=["m1"],
                topic="fastapi",
                text="Use uvicorn for FastAPI.",
                score=0.91,
                created_at="2024-01-01T00:00:00Z",
            )
        ]


def test_retriever_passes_filters_and_returns_contexts(tmp_path: Path) -> None:
    settings = Settings(
        hybrid_keyword=False,
        enable_rerank=False,
        processed_data_dir=tmp_path,
        top_k_default=5,
    )
    store = FakeStore()
    retriever = Retriever(FakeEmbedder(), store, settings)

    results = retriever.retrieve(
        question="How to run FastAPI?",
        top_k=3,
        topic="fastapi",
        date_from="2024-01-01T00:00:00Z",
        date_to="2024-12-31T00:00:00Z",
        chat_ids=["chat-1"],
    )

    assert len(results) == 1
    assert results[0].chat_id == "chat-1"
    assert store.last_kwargs["topic"] == "fastapi"
    assert store.last_kwargs["chat_ids"] == ["chat-1"]
    assert store.last_kwargs["top_k"] >= 3
