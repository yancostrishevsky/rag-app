"""Integration-style tests for the FastAPI app."""
import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VECTOR_STORE_DB_PATH", str(tmp_path / "store.sqlite3"))
    monkeypatch.setenv("VECTOR_STORE_EMBEDDING_DIM", "128")

    for module_name in (
        "vector_store_service.app",
        "vector_store_service.settings",
        "vector_store_service.vector_store",
    ):
        sys.modules.pop(module_name, None)

    return importlib.import_module("vector_store_service.app")


@pytest.fixture
def client(app_module):
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_documents_search_and_context(client: TestClient, unique_tokens) -> None:
    token_a, token_b, token_c = unique_tokens(3, 128)

    response = client.post(
        "/documents",
        json={
            "documents": [
                {
                    "doc_id": "doc-1",
                    "title": "Doc 1",
                    "content": f"{token_a} {token_b}"
                },
                {
                    "doc_id": "doc-2",
                    "title": "Doc 2",
                    "content": f"{token_c}"
                }
            ]
        }
    )
    assert response.status_code == 200
    assert response.json()["upserted"] == 2

    response = client.post(
        "/search",
        json={
            "query": f"{token_a} {token_b}",
            "top_k": 2
        }
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert results[0]["doc_id"] == "doc-1"

    response = client.post(
        "/collect_context_info",
        json={
            "user_message": f"{token_a} {token_b}",
            "chat_history": [],
            "top_k": 1
        }
    )
    assert response.status_code == 200
    context_docs = response.json()["context_docs"]
    assert context_docs
    assert context_docs[0][0] == "Doc 1"


def test_search_min_score_returns_empty(client: TestClient, unique_tokens) -> None:
    token_a = unique_tokens(1, 128)[0]
    client.post(
        "/documents",
        json={
            "documents": [
                {
                    "doc_id": "doc-1",
                    "title": "Doc 1",
                    "content": token_a
                }
            ]
        }
    )

    response = client.post(
        "/search",
        json={
            "query": token_a,
            "top_k": 3,
            "min_score": 1.1
        }
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_stream_chat_response_includes_titles(client: TestClient) -> None:
    response = client.post(
        "/stream_chat_response",
        json={
            "user_message": "Hello",
            "chat_history": [],
            "context_docs": [
                ["Doc A", "Content A"],
                ["Doc B", "Content B"]
            ]
        }
    )
    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert "Doc A" in body
    assert "Doc B" in body
