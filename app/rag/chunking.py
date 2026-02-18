from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import tiktoken

from app.rag.ingest.normalize import count_roles
from app.rag.schema import ChunkRecord, NormalizedMessage


def _encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_TOKENIZER = _encoding()


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    if _TOKENIZER is None:
        return max(1, len(text) // 4)
    return len(_TOKENIZER.encode(text))


def _message_to_chunk_line(message: NormalizedMessage) -> str:
    stamp = message.created_at or "unknown_time"
    return f"[{message.role} | {stamp} | {message.message_id}]\n{message.text}".strip()


def _timestamp_sort_key(value: str | None) -> tuple[int, str]:
    if not value:
        return (1, "")
    return (0, value)


def _choose_topic(messages: list[NormalizedMessage]) -> str:
    topics = [m.topic for m in messages if m.topic]
    if not topics:
        return "unknown"
    first = topics[0]
    return first


def build_chunks(
    messages: list[NormalizedMessage],
    max_tokens: int,
    overlap_messages: int,
) -> list[ChunkRecord]:
    grouped: dict[str, list[NormalizedMessage]] = defaultdict(list)
    for message in messages:
        grouped[message.chat_id].append(message)

    chunks: list[ChunkRecord] = []

    for _, chat_messages in grouped.items():
        chat_messages.sort(key=lambda m: (_timestamp_sort_key(m.created_at), m.message_id))

        current_entries: list[tuple[NormalizedMessage, str, int]] = []
        current_tokens = 0

        def emit(entries: list[tuple[NormalizedMessage, str, int]]) -> None:
            if not entries:
                return

            selected_messages = [entry[0] for entry in entries]
            text = "\n\n".join(entry[1] for entry in entries)
            chunks.append(
                ChunkRecord(
                    chunk_id=str(uuid4()),
                    chat_id=selected_messages[0].chat_id,
                    chat_title=selected_messages[0].chat_title,
                    message_ids=[m.message_id for m in selected_messages],
                    start_at=selected_messages[0].created_at,
                    end_at=selected_messages[-1].created_at,
                    topic=_choose_topic(selected_messages),
                    text=text,
                    metadata={
                        "title": selected_messages[0].chat_title,
                        "roles_count": count_roles(selected_messages),
                        "message_count": len(selected_messages),
                        "has_code": any(m.has_code for m in selected_messages),
                    },
                )
            )

        for message in chat_messages:
            chunk_line = _message_to_chunk_line(message)
            line_tokens = approx_token_count(chunk_line)

            if current_entries and current_tokens + line_tokens > max_tokens:
                emit(current_entries)
                if overlap_messages > 0:
                    current_entries = current_entries[-overlap_messages:].copy()
                    current_tokens = sum(item[2] for item in current_entries)
                else:
                    current_entries = []
                    current_tokens = 0

            if line_tokens > max_tokens:
                if current_entries:
                    # Avoid carrying overlap into an oversized standalone chunk.
                    current_entries = []
                    current_tokens = 0
                emit([(message, chunk_line, line_tokens)])
                current_entries = []
                current_tokens = 0
                continue

            if current_entries and current_tokens + line_tokens > max_tokens:
                current_entries = []
                current_tokens = 0

            current_entries.append((message, chunk_line, line_tokens))
            current_tokens += line_tokens

        if current_entries:
            emit(current_entries)

    return chunks


def write_chunks_jsonl(path: Path, chunks: list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json())
            f.write("\n")


def load_chunks_jsonl(path: Path) -> list[ChunkRecord]:
    if not path.exists():
        return []

    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunks.append(ChunkRecord.model_validate(data))
    return chunks
