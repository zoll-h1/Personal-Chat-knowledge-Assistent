from app.rag.ingest.redaction import redact_text


def test_redacts_email_phone_and_tokens() -> None:
    text = (
        "Contact me at dev@example.com or +1 415-555-1212. "
        "Key=sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAA and hf_BBBBBBBBBBBBBBBBBBBBBBBB"
    )
    redacted, stats = redact_text(text)

    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert redacted.count("[REDACTED_TOKEN]") >= 2
    assert stats.emails == 1
    assert stats.phones == 1
    assert stats.tokens >= 2


def test_redacts_password_patterns() -> None:
    text = "password=supersecret pwd:letmein"
    redacted, stats = redact_text(text)

    assert "[REDACTED_PASSWORD]" in redacted
    assert stats.passwords == 2
