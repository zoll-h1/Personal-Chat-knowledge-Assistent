from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from bs4 import BeautifulSoup

from app.rag.ingest.normalize import normalize_timestamp, role_from_raw
from app.rag.schema import NormalizedMessage


def parse_chatgpt_html(html_text: str, file_name: str) -> list[NormalizedMessage]:
    soup = BeautifulSoup(html_text, "html.parser")
    chat_id = Path(file_name).stem or f"html-chat-{uuid4()}"
    chat_title = soup.title.text.strip() if soup.title and soup.title.text else None

    role_nodes = soup.select("[data-message-author-role]")
    messages: list[NormalizedMessage] = []
    previous_id: str | None = None

    if role_nodes:
        for idx, node in enumerate(role_nodes):
            role = role_from_raw(node.attrs.get("data-message-author-role"))
            text = node.get_text("\n", strip=True)
            if not text:
                continue
            message_id = f"{chat_id}-html-{idx}"
            created_at = normalize_timestamp(node.attrs.get("data-message-created-at"))
            messages.append(
                NormalizedMessage(
                    chat_id=chat_id,
                    chat_title=chat_title,
                    message_id=message_id,
                    parent_message_id=previous_id,
                    role=role,
                    created_at=created_at,
                    text=text,
                    has_code=("```" in text or node.find("code") is not None),
                    attachments=[],
                    topic="unknown",
                    source="chatgpt_export_html",
                )
            )
            previous_id = message_id
        return messages

    candidate_nodes = soup.find_all(["article", "div", "p", "li"])
    for idx, node in enumerate(candidate_nodes):
        text = node.get_text("\n", strip=True)
        if not text:
            continue

        role = "unknown"
        lowered = text.lower()
        if lowered.startswith("user:"):
            role = "user"
            text = text[5:].strip()
        elif lowered.startswith("assistant:"):
            role = "assistant"
            text = text[10:].strip()
        elif lowered.startswith("system:"):
            role = "system"
            text = text[7:].strip()

        if not text:
            continue

        message_id = f"{chat_id}-fallback-{idx}"
        messages.append(
            NormalizedMessage(
                chat_id=chat_id,
                chat_title=chat_title,
                message_id=message_id,
                parent_message_id=previous_id,
                role=role,
                created_at=None,
                text=text,
                has_code=("```" in text or node.find("code") is not None),
                attachments=[],
                topic="unknown",
                source="chatgpt_export_html",
            )
        )
        previous_id = message_id

    return messages


def parse_chatgpt_html_bytes(raw: bytes, file_name: str) -> list[NormalizedMessage]:
    text = raw.decode("utf-8", errors="replace")
    return parse_chatgpt_html(text, file_name=file_name)
