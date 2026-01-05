"""Service configuration helpers."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""
    data_dir: Path
    db_path: Path
    index_path: Path
    embedding_dim: int
    embedder_type: str
    model_name: str
    top_k: int
    min_score: float


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value.strip() == '':
        return default
    return int(value)


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or value.strip() == '':
        return default
    return float(value)


def load_settings() -> Settings:
    """Loads settings from environment variables with sane defaults."""
    service_root = Path(__file__).resolve().parents[2]
    data_dir = Path(os.getenv('VECTOR_STORE_DATA_DIR', service_root / 'data'))
    db_path = Path(os.getenv('VECTOR_STORE_DB_PATH', data_dir / 'vector_store.sqlite3'))
    index_path = Path(os.getenv('VECTOR_STORE_INDEX_PATH', data_dir / 'faiss.index'))
    embedding_dim = _env_int('VECTOR_STORE_EMBEDDING_DIM', 384)
    embedder_type = os.getenv('VECTOR_STORE_EMBEDDER', 'sentence-transformers')
    model_name = os.getenv(
        'VECTOR_STORE_MODEL_NAME',
        'sentence-transformers/all-MiniLM-L6-v2'
    )
    top_k = _env_int('VECTOR_STORE_TOP_K', 5)
    min_score = _env_float('VECTOR_STORE_MIN_SCORE', 0.0)

    return Settings(
        data_dir=data_dir,
        db_path=db_path,
        index_path=index_path,
        embedding_dim=embedding_dim,
        embedder_type=embedder_type,
        model_name=model_name,
        top_k=top_k,
        min_score=min_score
    )
