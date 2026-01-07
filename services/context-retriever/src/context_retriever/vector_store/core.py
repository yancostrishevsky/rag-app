"""Contains core interfaces and utilities for vector store operations."""

from abc import ABC, abstractmethod
from typing import Iterator, TypeVar

import pydantic
from langchain_core.documents import Document


class EmbeddingModelCfg(pydantic.BaseModel):
    """Configuration of embedding model endpoint."""
    model_name: str
    url: str


class VectorStoreProxy(ABC):
    """Interface for classes handling vector store operations."""

    @abstractmethod
    def store_documents(self, documents: list[Document]) -> None:
        """Stores the given documents in the vector store.

        Args:
            documents: List of documents to be stored.
        """

    @abstractmethod
    def retrieve_documents(self, query: str) -> list[Document]:
        """Retrieves documents from the vector store based on the given query.

        Args:
            query: The query string to search for relevant documents.

        Returns:
            List of documents matching the query.
        """


T = TypeVar('T')


def batch_iterate(iterable: list[T], batch_size: int) -> Iterator[list[T]]:
    """Yields successive batches from an iterable.

    Args:
        iterable: The iterable to be divided into batches.
        batch_size: The size of each batch.
    """

    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]
