"""Tests for service settings loading."""
from pathlib import Path

from vector_store_service import settings as settings_module


def test_load_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("VECTOR_STORE_DATA_DIR", raising=False)
    monkeypatch.delenv("VECTOR_STORE_DB_PATH", raising=False)
    monkeypatch.delenv("VECTOR_STORE_EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("VECTOR_STORE_TOP_K", raising=False)
    monkeypatch.delenv("VECTOR_STORE_MIN_SCORE", raising=False)

    settings = settings_module.load_settings()
    service_root = Path(settings_module.__file__).resolve().parents[2]

    assert settings.data_dir == service_root / "data"
    assert settings.db_path == service_root / "data" / "vector_store.sqlite3"
    assert settings.embedding_dim == 256
    assert settings.top_k == 5
    assert settings.min_score == 0.0


def test_load_settings_from_env(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    db_path = tmp_path / "custom.sqlite3"

    monkeypatch.setenv("VECTOR_STORE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VECTOR_STORE_DB_PATH", str(db_path))
    monkeypatch.setenv("VECTOR_STORE_EMBEDDING_DIM", "64")
    monkeypatch.setenv("VECTOR_STORE_TOP_K", "7")
    monkeypatch.setenv("VECTOR_STORE_MIN_SCORE", "0.42")

    settings = settings_module.load_settings()

    assert settings.data_dir == data_dir
    assert settings.db_path == db_path
    assert settings.embedding_dim == 64
    assert settings.top_k == 7
    assert settings.min_score == 0.42
