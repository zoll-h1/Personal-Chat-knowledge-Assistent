from __future__ import annotations

from typing import Iterable

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class LocalEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, normalize_embeddings: bool = True) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        model = self._load_model()
        all_vectors: list[np.ndarray] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            logger.info("Embedding batch %s-%s/%s", start + 1, end, total)
            batch = texts[start:end]
            vectors = model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_vectors.append(vectors)

        return np.vstack(all_vectors).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.embed_texts([text])
        return vectors[0]

    def embedding_dimension(self) -> int:
        probe = self.embed_query("dimension_probe")
        return int(probe.shape[0])
