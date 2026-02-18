from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"
)
OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
HF_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-zA-Z0-9]{20,}\b")
AWS_KEY_PATTERN = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
JWT_PATTERN = re.compile(r"\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
PASSWORD_PATTERN = re.compile(r"(?i)\b(?:password|pwd)\s*[:=]\s*[^\s,;]+")


@dataclass
class RedactionStats:
    emails: int = 0
    phones: int = 0
    tokens: int = 0
    passwords: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "emails": self.emails,
            "phones": self.phones,
            "tokens": self.tokens,
            "passwords": self.passwords,
        }


def _subn(pattern: re.Pattern[str], repl: str, text: str) -> tuple[str, int]:
    return pattern.subn(repl, text)


def redact_text(text: str) -> tuple[str, RedactionStats]:
    stats = RedactionStats()
    redacted = text

    redacted, count = _subn(EMAIL_PATTERN, "[REDACTED_EMAIL]", redacted)
    stats.emails += count

    redacted, count = _subn(PHONE_PATTERN, "[REDACTED_PHONE]", redacted)
    stats.phones += count

    for token_pattern in (OPENAI_KEY_PATTERN, HF_TOKEN_PATTERN, AWS_KEY_PATTERN, JWT_PATTERN):
        redacted, count = _subn(token_pattern, "[REDACTED_TOKEN]", redacted)
        stats.tokens += count

    redacted, count = _subn(PASSWORD_PATTERN, "[REDACTED_PASSWORD]", redacted)
    stats.passwords += count

    return redacted, stats
