"""Unit tests for Web UI and REST API."""
import pytest
from src.web import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verify web UI loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Aster & Row" in response.data


def test_api_chat_route(client):
    """Verify /api/chat returns structured JSON with answer and citations."""
    payload = {"message": "How long do I have to return an unused backpack?"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "answer" in data
    assert "30" in data["answer"]
    assert len(data["sources"]) > 0
    assert any("01-returns-policy-current.md" in s for s in data["sources"])
