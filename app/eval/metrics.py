from __future__ import annotations

from statistics import mean


def hit_at_k(retrieved_chat_ids: list[str], expected_chat_id: str | None) -> float:
    if not expected_chat_id:
        return 0.0
    return 1.0 if expected_chat_id in retrieved_chat_ids else 0.0


def keyword_hit(retrieved_texts: list[str], expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    lowered_text = "\n".join(retrieved_texts).lower()
    return 1.0 if any(keyword.lower() in lowered_text for keyword in expected_keywords) else 0.0


def average_latency_ms(latencies_ms: list[float]) -> float:
    if not latencies_ms:
        return 0.0
    return float(mean(latencies_ms))


def abstain_rate(abstains: list[bool]) -> float:
    if not abstains:
        return 0.0
    return sum(1 for v in abstains if v) / len(abstains)
