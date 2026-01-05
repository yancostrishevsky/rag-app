"""Embedding providers for the vector store service."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import hashlib
import math
import re
from typing import Protocol

import numpy as np


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class Embedder(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns a 2D array of embeddings for the given texts."""


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'little') % dim


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = math.sqrt(float(np.dot(vector, vector)))
    if norm == 0:
        return vector
    return vector / norm


@dataclass
class HashingEmbedder:
    """Deterministic hashing embedder for testing and fallback."""
    embedding_dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for index, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                vectors[index, _hash_token(token, self.embedding_dim)] += 1.0
            vectors[index] = _normalize(vectors[index])
        return vectors


@dataclass
class SentenceTransformerEmbedder:
    """Sentence-transformers embedder with lazy model loading."""
    model_name: str
    normalize: bool = True
    _model: object | None = field(default=None, init=False, repr=False)

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return embeddings.astype(np.float32)


def create_embedder(embedder_type: str,
                    model_name: str,
                    embedding_dim: int) -> Embedder:
    """Factory for embedding providers."""
    normalized = embedder_type.strip().lower()
    if normalized in {"hash", "hashing"}:
        return HashingEmbedder(embedding_dim=embedding_dim)
    if normalized in {"sentence-transformers", "sentence_transformer", "st"}:
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"Unsupported embedder type: {embedder_type}")
