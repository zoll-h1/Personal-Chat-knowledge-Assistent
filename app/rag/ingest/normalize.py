from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from app.rag.schema import NormalizedMessage

IT_TOPICS = {
    "python",
    "backend",
    "fastapi",
    "django",
    "devops",
    "linux",
    "databases",
    "algorithms",
    "networking",
}

TOPIC_KEYWORDS: dict[str, set[str]] = {
    "python": {"python", "pip", "venv", "pandas", "numpy", "pytest"},
    "backend": {"backend", "api", "rest", "graphql", "microservice", "flask"},
    "fastapi": {"fastapi", "uvicorn", "pydantic", "asgi"},
    "django": {"django", "orm", "admin.py", "manage.py"},
    "devops": {"docker", "kubernetes", "ci", "cd", "terraform", "ansible"},
    "linux": {"linux", "bash", "shell", "systemd", "ubuntu", "debian"},
    "databases": {"database", "postgres", "mysql", "sqlite", "qdrant", "redis", "sql"},
    "algorithms": {"algorithm", "complexity", "big o", "graph", "dynamic programming"},
    "networking": {"network", "tcp", "udp", "http", "dns", "socket"},
}


def normalize_timestamp(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return normalize_timestamp(float(stripped))
        try:
            dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            return None

    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except (OSError, OverflowError, ValueError):
            return None

    return None


def role_from_raw(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"user", "assistant", "system", "tool"}:
        return role
    return "unknown"


def infer_topic(text: str) -> str:
    lowered = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "other"


def infer_chat_topic(messages: list[NormalizedMessage], chat_title: str | None) -> str:
    title_topic = infer_topic(chat_title or "")
    if title_topic != "other":
        return title_topic

    sample = "\n".join(m.text for m in messages[:12])
    return infer_topic(sample)


def apply_topics(messages: list[NormalizedMessage]) -> list[NormalizedMessage]:
    grouped: dict[str, list[NormalizedMessage]] = defaultdict(list)
    for msg in messages:
        grouped[msg.chat_id].append(msg)

    for _, msgs in grouped.items():
        topic = infer_chat_topic(msgs, msgs[0].chat_title if msgs else None)
        for msg in msgs:
            msg.topic = topic if topic in IT_TOPICS else (topic if topic == "other" else "unknown")
    return messages


def should_include_chat(
    chat_title: str | None,
    allowlist_it_only: bool,
    exclude_title_keywords: list[str],
) -> bool:
    title = (chat_title or "").lower()
    if any(keyword in title for keyword in exclude_title_keywords):
        return False
    if not allowlist_it_only:
        return True
    inferred = infer_topic(title)
    if inferred in IT_TOPICS:
        return True
    return False


def count_roles(messages: list[NormalizedMessage]) -> dict[str, int]:
    counts = Counter(msg.role for msg in messages)
    return dict(counts)
