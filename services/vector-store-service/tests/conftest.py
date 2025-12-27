"""Pytest configuration for vector-store-service."""
from pathlib import Path
import sys

import pytest


SERVICE_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = SERVICE_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def unique_tokens():
    """Provides a generator for tokens with unique hash buckets."""
    from vector_store_service import vector_store as vs

    def _factory(count: int, dim: int = 128) -> list[str]:
        tokens: list[str] = []
        used: set[int] = set()
        for index in range(1, 5000):
            token = f"token{index}"
            bucket = vs._hash_token(token, dim)
            if bucket in used:
                continue
            tokens.append(token)
            used.add(bucket)
            if len(tokens) == count:
                return tokens
        raise RuntimeError("Unable to generate unique tokens.")

    return _factory
