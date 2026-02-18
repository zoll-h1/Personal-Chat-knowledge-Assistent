from __future__ import annotations

import re
from typing import Sequence

from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.prompts import SYSTEM_PROMPT
from app.rag.schema import RetrievalContext

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{3,}")


def _keywords(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


def _short_snippet(text: str, max_len: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


class AnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _insufficient_context(self, contexts: Sequence[RetrievalContext]) -> tuple[str, float]:
        snippets = [f"- {_short_snippet(ctx.text)}" for ctx in contexts[:3]]
        details = "\n".join(snippets) if snippets else "- No matching snippets found."
        text = (
            "insufficient context. I do not have enough grounded evidence in your indexed chats to answer reliably. "
            "Please narrow the question (topic/date/chat) or provide more details.\n\nClosest snippets:\n"
            f"{details}"
        )
        confidence = max((ctx.score for ctx in contexts), default=0.0)
        return text, confidence

    def _extractive_answer(self, question: str, contexts: Sequence[RetrievalContext]) -> str:
        query_terms = _keywords(question)
        selected_lines: list[str] = []

        for idx, ctx in enumerate(contexts[:5], start=1):
            lines = [line.strip() for line in ctx.text.splitlines() if line.strip()]
            chosen = None
            for line in lines:
                if query_terms and query_terms & _keywords(line):
                    chosen = line
                    break
            if chosen is None and lines:
                chosen = lines[0]
            if chosen:
                selected_lines.append(f"{chosen} [C{idx}]")

        if not selected_lines:
            selected_lines = [f"{_short_snippet(contexts[0].text)} [C1]"]

        return "\n".join(selected_lines)

    def _llm_answer(self, question: str, contexts: Sequence[RetrievalContext]) -> str:
        if not self.settings.openai_api_key:
            return self._extractive_answer(question, contexts)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.settings.openai_api_key)
            context_text = "\n\n".join(
                f"[C{idx}] score={ctx.score:.3f}\n{ctx.text}" for idx, ctx in enumerate(contexts[:8], start=1)
            )
            completion = client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nContext:\n{context_text}",
                    },
                ],
            )
            answer = completion.choices[0].message.content or "Insufficient context"
            return answer.strip()
        except Exception as exc:
            logger.warning("LLM mode failed, falling back to extractive mode: %s", exc)
            return self._extractive_answer(question, contexts)

    def generate(
        self,
        question: str,
        contexts: list[RetrievalContext],
        mode: str | None = None,
    ) -> tuple[str, float]:
        if not contexts:
            return self._insufficient_context(contexts)

        confidence = max(ctx.score for ctx in contexts)
        if confidence < self.settings.confidence_threshold:
            return self._insufficient_context(contexts)

        chosen_mode = mode or self.settings.mode
        if chosen_mode == "llm":
            return self._llm_answer(question, contexts), confidence

        return self._extractive_answer(question, contexts), confidence
