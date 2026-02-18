from fastapi.testclient import TestClient

from app.api.main import app
from app.api import routes_chat
from app.rag.schema import AskResponse, Citation


class FakeChatService:
    def health(self):
        return {"status": "ok"}

    def ask(self, request):
        return AskResponse(
            answer="Test answer",
            citations=[
                Citation(
                    chat_id="chat-1",
                    message_ids=["m1"],
                    snippet="sample",
                    score=0.9,
                    created_at="2024-01-01T00:00:00Z",
                )
            ],
            confidence=0.9,
            latency_ms=1.0,
        )


def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(routes_chat, "get_chat_service", lambda: FakeChatService())
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(routes_chat, "get_chat_service", lambda: FakeChatService())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "hello", "top_k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Test answer"
    assert payload["citations"][0]["chat_id"] == "chat-1"
