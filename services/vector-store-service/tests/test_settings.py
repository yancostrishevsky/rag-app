"""Tests for service settings loading."""
from pathlib import Path

from vector_store_service import settings as settings_module


def test_load_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("VECTOR_STORE_DATA_DIR", raising=False)
    monkeypatch.delenv("VECTOR_STORE_DB_PATH", raising=False)
    monkeypatch.delenv("VECTOR_STORE_INDEX_PATH", raising=False)
    monkeypatch.delenv("VECTOR_STORE_EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("VECTOR_STORE_EMBEDDER", raising=False)
    monkeypatch.delenv("VECTOR_STORE_MODEL_NAME", raising=False)
    monkeypatch.delenv("VECTOR_STORE_TOP_K", raising=False)
    monkeypatch.delenv("VECTOR_STORE_MIN_SCORE", raising=False)

    settings = settings_module.load_settings()
    service_root = Path(settings_module.__file__).resolve().parents[2]

    assert settings.data_dir == service_root / "data"
    assert settings.db_path == service_root / "data" / "vector_store.sqlite3"
    assert settings.index_path == service_root / "data" / "faiss.index"
    assert settings.embedding_dim == 384
    assert settings.embedder_type == "sentence-transformers"
    assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.top_k == 5
    assert settings.min_score == 0.0


def test_load_settings_from_env(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    db_path = tmp_path / "custom.sqlite3"
    index_path = tmp_path / "custom.index"

    monkeypatch.setenv("VECTOR_STORE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VECTOR_STORE_DB_PATH", str(db_path))
    monkeypatch.setenv("VECTOR_STORE_INDEX_PATH", str(index_path))
    monkeypatch.setenv("VECTOR_STORE_EMBEDDING_DIM", "64")
    monkeypatch.setenv("VECTOR_STORE_EMBEDDER", "hash")
    monkeypatch.setenv("VECTOR_STORE_MODEL_NAME", "local/model")
    monkeypatch.setenv("VECTOR_STORE_TOP_K", "7")
    monkeypatch.setenv("VECTOR_STORE_MIN_SCORE", "0.42")

    settings = settings_module.load_settings()

    assert settings.data_dir == data_dir
    assert settings.db_path == db_path
    assert settings.index_path == index_path
    assert settings.embedding_dim == 64
    assert settings.embedder_type == "hash"
    assert settings.model_name == "local/model"
    assert settings.top_k == 7
    assert settings.min_score == 0.42
