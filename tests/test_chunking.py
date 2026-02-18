from app.rag.chunking import build_chunks
from app.rag.schema import NormalizedMessage


def test_chunking_keeps_code_block_intact_and_overlaps() -> None:
    messages = [
        NormalizedMessage(
            chat_id="chat-1",
            chat_title="test",
            message_id="m1",
            parent_message_id=None,
            role="user",
            created_at="2024-01-01T00:00:00Z",
            text="Please provide SQL sample and explanation.",
            has_code=False,
            attachments=[],
            topic="databases",
            source="chatgpt_export_json",
        ),
        NormalizedMessage(
            chat_id="chat-1",
            chat_title="test",
            message_id="m2",
            parent_message_id="m1",
            role="assistant",
            created_at="2024-01-01T00:00:10Z",
            text="```sql\nSELECT * FROM users WHERE id = 1;\n```",
            has_code=True,
            attachments=[],
            topic="databases",
            source="chatgpt_export_json",
        ),
        NormalizedMessage(
            chat_id="chat-1",
            chat_title="test",
            message_id="m3",
            parent_message_id="m2",
            role="assistant",
            created_at="2024-01-01T00:00:20Z",
            text="Use an index on user id.",
            has_code=False,
            attachments=[],
            topic="databases",
            source="chatgpt_export_json",
        ),
    ]

    chunks = build_chunks(messages, max_tokens=25, overlap_messages=1)

    assert len(chunks) >= 2
    assert any("```sql\nSELECT * FROM users WHERE id = 1;\n```" in chunk.text for chunk in chunks)
    assert any("m2" in chunk.message_ids for chunk in chunks)
    assert any("m1" in chunk.message_ids for chunk in chunks[1:])
