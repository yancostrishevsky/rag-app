"""Tests for the SQLite-backed vector store."""
from pathlib import Path

import numpy as np

from vector_store_service.embeddings import HashingEmbedder
from vector_store_service.vector_store import Document
from vector_store_service.vector_store import VectorStore


def test_upsert_and_search_ranks_results(tmp_path: Path, unique_tokens) -> None:
    dim = 128
    token_a, token_b, token_c, title_token = unique_tokens(4, dim)
    embedder = HashingEmbedder(embedding_dim=dim)
    store = VectorStore(
        str(tmp_path / "store.sqlite3"),
        embedding_dim=dim,
        embedder=embedder
    )

    documents = [
        Document(doc_id="doc1", title=title_token, content=f"{token_a} {token_b}"),
        Document(doc_id="doc2", title=title_token, content=f"{token_a}"),
        Document(doc_id="doc3", title=title_token, content=f"{token_c}")
    ]
    store.upsert_documents(documents)

    query = f"{token_a} {token_b}"
    results = store.search(query, top_k=3)

    query_embedding = embedder.embed([query])[0]
    expected_scores = []
    for document in documents:
        doc_embedding = embedder.embed(
            [f"{document.title}\n{document.content}"]
        )[0]
        expected_scores.append(
            (document.doc_id, float(np.dot(query_embedding, doc_embedding)))
        )
    expected_scores.sort(key=lambda item: item[1], reverse=True)

    assert [doc.doc_id for doc, _ in results] == [
        doc_id for doc_id, _ in expected_scores
    ]


def test_upsert_updates_existing_document(tmp_path: Path, unique_tokens) -> None:
    dim = 128
    token_a, token_b, title_token = unique_tokens(3, dim)
    store = VectorStore(
        str(tmp_path / "store.sqlite3"),
        embedding_dim=dim,
        embedder=HashingEmbedder(embedding_dim=dim)
    )

    store.upsert_documents([
        Document(doc_id="doc1", title=title_token, content=token_a)
    ])
    store.upsert_documents([
        Document(doc_id="doc1", title=title_token, content=token_b)
    ])

    results = store.search(token_b, top_k=1)
    assert results
    assert results[0][0].content == token_b


def test_search_min_score_filters_results(tmp_path: Path, unique_tokens) -> None:
    dim = 128
    token_a, title_token = unique_tokens(2, dim)
    store = VectorStore(
        str(tmp_path / "store.sqlite3"),
        embedding_dim=dim,
        embedder=HashingEmbedder(embedding_dim=dim)
    )

    store.upsert_documents([
        Document(doc_id="doc1", title=title_token, content=token_a)
    ])

    results = store.search(token_a, top_k=1, min_score=1.1)
    assert results == []


def test_search_empty_query_returns_empty(tmp_path: Path) -> None:
    store = VectorStore(
        str(tmp_path / "store.sqlite3"),
        embedding_dim=64,
        embedder=HashingEmbedder(embedding_dim=64)
    )
    store.upsert_documents([
        Document(doc_id="doc1", title="Title", content="alpha beta")
    ])

    assert store.search("   ", top_k=3) == []
