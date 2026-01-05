"""SQLite-backed vector store with FAISS indexing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import threading
from typing import Iterable
from typing import List
from typing import Tuple

import faiss
import numpy as np

from vector_store_service.embeddings import Embedder


@dataclass(frozen=True)
class Document:
    """Simple document representation."""
    doc_id: str
    title: str
    content: str


class VectorStore:
    """SQLite-backed document store with FAISS vector indexing."""

    def __init__(self,
                 db_path: str,
                 embedding_dim: int,
                 embedder: Embedder,
                 index_path: str | None = None) -> None:
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._embedder = embedder
        self._index_path = index_path
        self._lock = threading.Lock()

        self._ensure_database()
        self._index = self._build_index()

    def _ensure_database(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _build_index(self) -> faiss.IndexIDMap2:
        cached_index = self._load_index_from_disk()
        if cached_index is not None:
            return cached_index

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._embedding_dim))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, embedding FROM documents"
            ).fetchall()

        if not rows:
            return index

        ids = np.array([row["id"] for row in rows], dtype="int64")
        vectors = np.vstack([
            self._deserialize_embedding(row["embedding"])
            for row in rows
        ])
        index.add_with_ids(vectors.astype(np.float32), ids)
        self._persist_index(index)
        return index

    def _serialize_embedding(self, vector: np.ndarray) -> bytes:
        return vector.astype(np.float32).tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != self._embedding_dim:
            raise ValueError("Stored embedding dimension does not match configuration.")
        return vector

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self._embedder.embed(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._embedding_dim:
            raise ValueError("Embedding dimension does not match configuration.")
        return vectors

    def _remove_from_index(self, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        selector = faiss.IDSelectorBatch(ids)
        self._index.remove_ids(selector)

    def _persist_index(self, index: faiss.IndexIDMap2 | None = None) -> None:
        if not self._index_path:
            return
        path = Path(self._index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index or self._index, str(path))

    def _load_index_from_disk(self) -> faiss.IndexIDMap2 | None:
        if not self._index_path:
            return None
        path = Path(self._index_path)
        if not path.exists():
            return None
        index = faiss.read_index(str(path))
        if not isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            return None
        if index.d != self._embedding_dim:
            return None

        if index.ntotal != self._count_documents():
            return None
        return index

    def _count_documents(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])

    def upsert_documents(self, documents: Iterable[Document]) -> List[str]:
        """Adds or updates documents and returns their ids."""
        documents_list = list(documents)
        if not documents_list:
            return []

        texts = [
            f"{document.title}\n{document.content}"
            for document in documents_list
        ]
        embeddings = self._embed_texts(texts)

        with self._lock, self._connect() as connection:
            doc_ids = [document.doc_id for document in documents_list]
            existing = {}
            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                rows = connection.execute(
                    f"SELECT id, doc_id FROM documents WHERE doc_id IN ({placeholders})",
                    doc_ids
                ).fetchall()
                existing = {row["doc_id"]: row["id"] for row in rows}

            if existing:
                existing_ids = np.array(list(existing.values()), dtype="int64")
                self._remove_from_index(existing_ids)

            index_ids: list[int] = []
            for document, embedding in zip(documents_list, embeddings):
                blob = self._serialize_embedding(embedding)
                if document.doc_id in existing:
                    doc_pk = existing[document.doc_id]
                    connection.execute(
                        """
                        UPDATE documents
                        SET title = ?, content = ?, embedding = ?
                        WHERE id = ?
                        """,
                        (document.title, document.content, blob, doc_pk)
                    )
                else:
                    cursor = connection.execute(
                        """
                        INSERT INTO documents (doc_id, title, content, embedding)
                        VALUES (?, ?, ?, ?)
                        """,
                        (document.doc_id, document.title, document.content, blob)
                    )
                    doc_pk = cursor.lastrowid
                index_ids.append(int(doc_pk))

            connection.commit()
            self._index.add_with_ids(
                embeddings.astype(np.float32),
                np.array(index_ids, dtype="int64")
            )
            self._persist_index()

        return [document.doc_id for document in documents_list]

    def search(self,
               query: str,
               top_k: int = 5,
               min_score: float = 0.0) -> List[Tuple[Document, float]]:
        """Returns top-k documents by cosine similarity."""
        if not query.strip() or top_k <= 0:
            return []

        with self._lock:
            if self._index.ntotal == 0:
                return []

        query_vector = self._embed_texts([query])

        with self._lock:
            scores, ids = self._index.search(query_vector, top_k)
            pairs = [
                (int(doc_id), float(score))
                for doc_id, score in zip(ids[0], scores[0])
                if doc_id != -1 and score >= min_score
            ]

            if not pairs:
                return []

            documents = self._fetch_documents_by_ids([doc_id for doc_id, _ in pairs])

        results: List[Tuple[Document, float]] = []
        for doc_id, score in pairs:
            document = documents.get(doc_id)
            if document is not None:
                results.append((document, score))
        return results

    def _fetch_documents_by_ids(self, ids: list[int]) -> dict[int, Document]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, doc_id, title, content
                FROM documents
                WHERE id IN ({placeholders})
                """,
                ids
            ).fetchall()

        return {
            int(row["id"]): Document(
                doc_id=row["doc_id"],
                title=row["title"],
                content=row["content"]
            )
            for row in rows
        }
