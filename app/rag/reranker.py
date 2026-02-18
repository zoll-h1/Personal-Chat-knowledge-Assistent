from __future__ import annotations

import re

from app.rag.schema import RetrievalContext


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{3,}")


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


class LexicalReranker:
    def rerank(self, query: str, contexts: list[RetrievalContext], top_k: int) -> list[RetrievalContext]:
        if not contexts:
            return contexts

        query_terms = _tokenize(query)
        rescored: list[RetrievalContext] = []

        for ctx in contexts:
            doc_terms = _tokenize(ctx.text)
            if not query_terms or not doc_terms:
                overlap_score = 0.0
            else:
                overlap_score = len(query_terms & doc_terms) / len(query_terms)
            boosted = min(1.0, ctx.score + 0.15 * overlap_score)
            rescored.append(ctx.model_copy(update={"score": boosted}))

        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_k]
