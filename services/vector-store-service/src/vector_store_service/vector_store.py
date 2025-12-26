"""Lightweight SQLite-backed vector store for document retrieval."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Document:
    """Simple document representation."""
    doc_id: str
    title: str
    content: str


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'little') % dim


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


def _embed_text(text: str, dim: int) -> List[float]:
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * dim
    counts = [0.0] * dim
    for token in tokens:
        counts[_hash_token(token, dim)] += 1.0
    return _normalize(counts)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(l * r for l, r in zip(left, right))


class VectorStore:
    """SQLite-backed vector store with hashed bag-of-words embeddings."""

    def __init__(self, db_path: str, embedding_dim: int = 256) -> None:
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._ensure_database()

    def _ensure_database(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def upsert_documents(self, documents: Iterable[Document]) -> List[str]:
        """Adds or updates documents and returns their ids."""
        doc_ids: List[str] = []
        with self._connect() as connection:
            for document in documents:
                embedding = _embed_text(
                    f"{document.title}\n{document.content}",
                    self._embedding_dim
                )
                connection.execute(
                    """
                    INSERT INTO documents (doc_id, title, content, embedding)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        title=excluded.title,
                        content=excluded.content,
                        embedding=excluded.embedding
                    """,
                    (
                        document.doc_id,
                        document.title,
                        document.content,
                        json.dumps(embedding)
                    )
                )
                doc_ids.append(document.doc_id)
            connection.commit()
        return doc_ids

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.0) -> List[Tuple[Document, float]]:
        """Returns top-k documents by cosine similarity."""
        query_embedding = _embed_text(query, self._embedding_dim)
        if not any(query_embedding):
            return []

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT doc_id, title, content, embedding FROM documents"
            ).fetchall()

        scored: List[Tuple[Document, float]] = []
        for row in rows:
            embedding = json.loads(row['embedding'])
            score = _dot(query_embedding, embedding)
            if score < min_score:
                continue
            scored.append((
                Document(
                    doc_id=row['doc_id'],
                    title=row['title'],
                    content=row['content']
                ),
                score
            ))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
