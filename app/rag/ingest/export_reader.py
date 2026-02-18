from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.rag.ingest.normalize import IT_TOPICS, apply_topics
from app.rag.ingest.parser_chatgpt_html import parse_chatgpt_html_bytes
from app.rag.ingest.parser_chatgpt_json import parse_chatgpt_json_bytes
from app.rag.ingest.redaction import RedactionStats, redact_text
from app.rag.schema import NormalizedMessage

logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {".zip", ".json", ".html", ".htm"}


def _read_zip(path: Path) -> list[NormalizedMessage]:
    messages: list[NormalizedMessage] = []
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            lowered = name.lower()
            if lowered.endswith(".json"):
                with zf.open(name) as fp:
                    raw = fp.read()
                try:
                    parsed = parse_chatgpt_json_bytes(raw)
                    messages.extend(parsed)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON inside ZIP: %s", name)
            elif lowered.endswith(".html") or lowered.endswith(".htm"):
                with zf.open(name) as fp:
                    raw = fp.read()
                messages.extend(parse_chatgpt_html_bytes(raw, file_name=name))
    return messages


def _read_file(path: Path) -> list[NormalizedMessage]:
    lowered = path.suffix.lower()
    raw = path.read_bytes()

    if lowered == ".json":
        return parse_chatgpt_json_bytes(raw)
    if lowered in {".html", ".htm"}:
        return parse_chatgpt_html_bytes(raw, file_name=path.name)
    if lowered == ".zip":
        return _read_zip(path)
    raise ValueError(f"Unsupported input type: {path.suffix}")


def resolve_input_path(raw_data_dir: Path, user_input_path: str | None) -> Path:
    if user_input_path:
        candidate = Path(user_input_path)
        if candidate.exists():
            return candidate
        raw_candidate = raw_data_dir / user_input_path
        if raw_candidate.exists():
            return raw_candidate
        raise FileNotFoundError(f"Input file not found: {user_input_path}")

    options = [p for p in raw_data_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not options:
        raise FileNotFoundError(
            f"No supported export files found in {raw_data_dir}. Add a ZIP/JSON/HTML file first."
        )

    options.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return options[0]


def _write_jsonl(path: Path, records: list[NormalizedMessage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def _apply_privacy(messages: list[NormalizedMessage]) -> tuple[list[NormalizedMessage], RedactionStats]:
    total_stats = RedactionStats()
    cleaned: list[NormalizedMessage] = []

    for message in messages:
        redacted_text, stats = redact_text(message.text)
        message.text = redacted_text
        message.has_code = "```" in redacted_text

        total_stats.emails += stats.emails
        total_stats.phones += stats.phones
        total_stats.tokens += stats.tokens
        total_stats.passwords += stats.passwords

        # Skip null/empty messages safely after normalization/redaction.
        if not message.text.strip() and not message.attachments:
            continue

        cleaned.append(message)

    return cleaned, total_stats


def _filter_messages(
    messages: list[NormalizedMessage],
    allowlist_it_only: bool,
    exclude_title_keywords: list[str],
) -> list[NormalizedMessage]:
    grouped: dict[str, list[NormalizedMessage]] = defaultdict(list)
    for message in messages:
        grouped[message.chat_id].append(message)

    filtered: list[NormalizedMessage] = []
    for chat_id, chat_messages in grouped.items():
        title = (chat_messages[0].chat_title or "").lower()
        if any(keyword in title for keyword in exclude_title_keywords):
            continue

        topic = chat_messages[0].topic
        if allowlist_it_only and topic not in IT_TOPICS:
            continue

        filtered.extend(chat_messages)

    return filtered


def ingest_export(
    input_path: Path,
    output_messages_path: Path,
    allowlist_it_only: bool,
    exclude_title_keywords: list[str],
) -> dict[str, Any]:
    logger.info("Reading export from %s", input_path)
    messages = _read_file(input_path)
    raw_count = len(messages)

    messages, redaction_stats = _apply_privacy(messages)
    messages = apply_topics(messages)
    messages = _filter_messages(
        messages,
        allowlist_it_only=allowlist_it_only,
        exclude_title_keywords=[k.lower() for k in exclude_title_keywords],
    )

    _write_jsonl(output_messages_path, messages)
    logger.info("Wrote %s normalized messages to %s", len(messages), output_messages_path)

    unique_chats = {m.chat_id for m in messages}
    return {
        "input_path": str(input_path),
        "raw_message_count": raw_count,
        "processed_message_count": len(messages),
        "chat_count": len(unique_chats),
        "redaction": redaction_stats.to_dict(),
        "output_messages_path": str(output_messages_path),
    }


def load_messages_jsonl(path: Path) -> list[NormalizedMessage]:
    if not path.exists():
        return []

    messages: list[NormalizedMessage] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            messages.append(NormalizedMessage.model_validate(data))
    return messages
