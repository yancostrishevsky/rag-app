"""Contains class for handling communication with Qdrant vector store."""

import logging

import qdrant_client
from qdrant_client.models import VectorParams, Distance
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import pydantic

from context_retriever.vector_store import core as vs_core


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class QdrantProxyCfg(pydantic.BaseModel):
    """Configuration for vector store proxy."""
    url: str
    collection_name: str
    uploading_batch_size: int
    embedding_model: vs_core.EmbeddingModelCfg


class QdrantProxy(vs_core.VectorStoreProxy):
    """Handles communication with Qdrant vector store."""

    def __init__(self, cfg: QdrantProxyCfg) -> None:

        _logger().info('Initializing QdrantProxy with config: %s', cfg)

        self._cfg = cfg

        self._client = qdrant_client.QdrantClient(
            url=self._cfg.url
        )

        self._embedding_model = OllamaEmbeddings(
            model=self._cfg.embedding_model.model_name,
            base_url=self._cfg.embedding_model.url
        )

        self._ensure_collection_exists()

    def store_documents(self, documents: list[Document]) -> None:
        """Stores the given documents in the vector store.

        Args:
            documents: List of documents to be stored.
        """

        _logger().debug('Storing %d documents in collection \'%s\'',
                        len(documents), self._cfg.collection_name)

        for batch_docs in vs_core.batch_iterate(documents, self._cfg.uploading_batch_size):

            embedded_docs = self._embedding_model.embed_documents(
                [doc.page_content for doc in batch_docs])

            self._client.upload_collection(
                collection_name=self._cfg.collection_name,
                vectors=embedded_docs,
                payload=[{'text': doc.page_content} for doc in batch_docs],
            )

    def retrieve_documents(self, query: str) -> list[Document]:
        """Retrieves documents from the vector store based on the given query.

        Args:
            query: The query string to search for relevant documents.

        Returns:
            List of documents matching the query.
        """

        _logger().debug('Retrieving documents from collection \'%s\' for query: \'%s\'',
                        self._cfg.collection_name, query)

        return []

    def _ensure_collection_exists(self) -> None:
        """Ensures that the specified collection exists in the Qdrant vector store."""

        if not self._client.collection_exists(self._cfg.collection_name):

            _logger().debug('Collection \'%s\' does not exist. Creating new collection.',
                            self._cfg.collection_name)

            vector_size = self._determine_vector_size()

            self._client.create_collection(
                collection_name=self._cfg.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.DOT, on_disk=True)
            )

    def _determine_vector_size(self) -> int:
        """Determines the size of the embedding vectors produced by the embedding model."""

        sample_text = "Sample text for embedding size determination."
        embedding = self._embedding_model.embed_query(sample_text)
        return len(embedding)
