from app.rag.ingest.parser_chatgpt_json import parse_chatgpt_json


def test_mapping_parent_is_resolved_to_parent_message_id() -> None:
    payload = {
        "conversations": [
            {
                "id": "chat-1",
                "title": "test",
                "mapping": {
                    "node-a": {
                        "id": "node-a",
                        "message": {
                            "id": "msg-a",
                            "author": {"role": "user"},
                            "content": {"parts": ["hello"]},
                        },
                    },
                    "node-b": {
                        "id": "node-b",
                        "parent": "node-a",
                        "message": {
                            "id": "msg-b",
                            "author": {"role": "assistant"},
                            "content": {"parts": ["hi"]},
                        },
                    },
                },
            }
        ]
    }

    records = parse_chatgpt_json(payload)
    by_id = {r.message_id: r for r in records}

    assert "msg-b" in by_id
    assert by_id["msg-b"].parent_message_id == "msg-a"
