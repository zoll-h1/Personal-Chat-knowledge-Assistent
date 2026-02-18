from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from app.rag.ingest.normalize import normalize_timestamp, role_from_raw
from app.rag.schema import NormalizedMessage


def _extract_conversations(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("conversations"), list):
            return [item for item in payload["conversations"] if isinstance(item, dict)]
        if isinstance(payload.get("items"), list):
            return [item for item in payload["items"] if isinstance(item, dict)]
        if any(key in payload for key in ("mapping", "messages", "id", "conversation_id")):
            return [payload]
    return []


def _extract_attachments(message_obj: dict[str, Any]) -> list[dict[str, Any]]:
    attachments: list[dict[str, Any]] = []
    metadata = message_obj.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("attachments"), list):
        for item in metadata.get("attachments", []):
            if isinstance(item, dict):
                attachments.append(
                    {
                        "type": item.get("mime_type") or item.get("type") or "attachment",
                        "name": item.get("name"),
                        "caption": item.get("caption") or item.get("text"),
                    }
                )

    content = message_obj.get("content")
    if isinstance(content, dict) and isinstance(content.get("parts"), list):
        for part in content["parts"]:
            if isinstance(part, dict):
                part_type = str(part.get("content_type") or part.get("type") or "text")
                if part_type != "text":
                    attachments.append(
                        {
                            "type": part_type,
                            "name": part.get("name"),
                            "caption": part.get("caption") or part.get("text"),
                        }
                    )
    return attachments


def _extract_text(message_obj: dict[str, Any]) -> str:
    chunks: list[str] = []

    content = message_obj.get("content")
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, str):
                    if part.strip():
                        chunks.append(part)
                elif isinstance(part, dict):
                    if isinstance(part.get("text"), str) and part["text"].strip():
                        chunks.append(part["text"])
                    elif isinstance(part.get("caption"), str) and part["caption"].strip():
                        chunks.append(part["caption"])
        elif isinstance(content.get("text"), str):
            chunks.append(content["text"])

    if isinstance(message_obj.get("text"), str):
        chunks.append(message_obj["text"])

    if isinstance(message_obj.get("parts"), list):
        for part in message_obj["parts"]:
            if isinstance(part, str) and part.strip():
                chunks.append(part)

    if isinstance(message_obj.get("body"), str):
        chunks.append(message_obj["body"])

    return "\n".join(item for item in chunks if item).strip()


def _node_to_message(
    chat_id: str,
    chat_title: str | None,
    source_obj: dict[str, Any],
    node_id: str,
    parent_id: str | None,
    message_id: str | None = None,
) -> NormalizedMessage | None:
    message_obj = source_obj.get("message") if isinstance(source_obj.get("message"), dict) else source_obj

    text = _extract_text(message_obj)
    attachments = _extract_attachments(message_obj)
    if not text and not attachments:
        return None

    author = message_obj.get("author")
    role = "unknown"
    if isinstance(author, dict):
        role = role_from_raw(author.get("role"))
    if role == "unknown":
        role = role_from_raw(message_obj.get("role") or source_obj.get("role"))

    created_raw = (
        message_obj.get("create_time")
        or source_obj.get("create_time")
        or source_obj.get("created_at")
    )

    msg_id = str(message_id or message_obj.get("id") or source_obj.get("id") or node_id or uuid4())

    return NormalizedMessage(
        chat_id=chat_id,
        chat_title=chat_title,
        message_id=msg_id,
        parent_message_id=parent_id,
        role=role,
        created_at=normalize_timestamp(created_raw),
        text=text,
        has_code="```" in text,
        attachments=attachments,
        topic="unknown",
        source="chatgpt_export_json",
    )


def _parse_mapping(
    mapping: dict[str, Any],
    chat_id: str,
    chat_title: str | None,
) -> list[NormalizedMessage]:
    # Map node IDs to actual message IDs first so parent_message_id references
    # stay stable across branching/regenerated paths.
    node_to_message_id: dict[str, str] = {}
    for key, node in mapping.items():
        if not isinstance(node, dict):
            continue
        message_obj = node.get("message") if isinstance(node.get("message"), dict) else node
        resolved_id = message_obj.get("id") or node.get("id") or f"{chat_id}-node-{key}"
        node_to_message_id[str(key)] = str(resolved_id)

    messages: list[NormalizedMessage] = []
    for key, node in mapping.items():
        if not isinstance(node, dict):
            continue
        parent_node_id = node.get("parent") or node.get("parent_id")
        parent_id = None
        if parent_node_id:
            parent_id = node_to_message_id.get(str(parent_node_id), str(parent_node_id))

        parsed = _node_to_message(
            chat_id=chat_id,
            chat_title=chat_title,
            source_obj=node,
            node_id=str(key),
            parent_id=parent_id,
            message_id=node_to_message_id.get(str(key)),
        )
        if parsed:
            messages.append(parsed)
    return messages


def _parse_messages_list(
    raw_messages: list[Any],
    chat_id: str,
    chat_title: str | None,
) -> list[NormalizedMessage]:
    messages: list[NormalizedMessage] = []
    previous_id: str | None = None

    for idx, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("id") or item.get("message_id") or f"{chat_id}-m-{idx}")
        parent = item.get("parent_message_id") or item.get("parent") or previous_id
        parsed = _node_to_message(chat_id, chat_title, item, node_id, str(parent) if parent else None)
        if parsed:
            messages.append(parsed)
            previous_id = parsed.message_id

    return messages


def parse_chatgpt_json(payload: Any) -> list[NormalizedMessage]:
    conversations = _extract_conversations(payload)
    output: list[NormalizedMessage] = []

    for idx, convo in enumerate(conversations):
        chat_id = str(convo.get("id") or convo.get("conversation_id") or f"chat-{idx}-{uuid4()}")
        chat_title = convo.get("title") if isinstance(convo.get("title"), str) else None

        mapping = convo.get("mapping")
        if isinstance(mapping, dict):
            output.extend(_parse_mapping(mapping, chat_id, chat_title))
            continue

        raw_messages = convo.get("messages")
        if isinstance(raw_messages, list):
            output.extend(_parse_messages_list(raw_messages, chat_id, chat_title))
            continue

        if isinstance(convo.get("conversation"), dict):
            nested = convo["conversation"]
            output.extend(parse_chatgpt_json(nested))

    return output


def parse_chatgpt_json_bytes(raw: bytes) -> list[NormalizedMessage]:
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    return parse_chatgpt_json(payload)
