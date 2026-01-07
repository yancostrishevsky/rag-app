"""Contains service that prepares documents for storage in the vector store."""

from typing import BinaryIO
import tempfile
import logging

import pydantic
from langchain_community import document_loaders
import langchain_text_splitters
from langchain_core.documents import Document

from context_retriever.vector_store.core import VectorStoreProxy


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class DocProcessingCfg(pydantic.BaseModel):
    """Configuration for document processing."""
    chunk_size: int
    chunk_overlap: int


class DocPreparationService:
    """Validates, processes and saves documents to the vector store."""

    _MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
    _PDF_ALLOWED_FIELDS = ('title', 'author')

    def __init__(self,
                 vector_store_proxy: VectorStoreProxy,
                 doc_processing_cfg: DocProcessingCfg) -> None:

        _logger().info('Initializing DocPreparationService with config: %s',
                       doc_processing_cfg)

        self._vector_store_proxy = vector_store_proxy

        self._text_splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=doc_processing_cfg.chunk_size,
            chunk_overlap=doc_processing_cfg.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def upload_pdf(self,
                   file_size: int,
                   content_type: str,
                   file_stream: BinaryIO
                   ) -> tuple[bool, str]:
        """Uploads a PDF document to the vector store.
        Args:
            file_size: Size of the file in bytes.
            content_type: Content type of the file.
            file_stream: Stream of the file to be uploaded.

        Returns:
            A tuple where the first element indicates success, and the second element contains
            an error message if any.
        """

        if content_type != 'application/pdf':
            return False, 'Invalid content type. Only PDF files are supported.'

        if file_size > self._MAX_PDF_SIZE_BYTES:
            return (False,
                    f'File size exceeds the maximum limit of {self._MAX_PDF_SIZE_BYTES} bytes.')

        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(file_stream.read())
            temp_file.flush()

            pdf_pages = document_loaders.PyPDFLoader(temp_file.name).load()

            self._sanitize_pdf_metadata(pdf_pages)

        _logger().debug('Uploading PDF: size=%d, title=%s.',
                        file_size,
                        pdf_pages[0].metadata['title'])

        chunks = self._text_splitter.split_documents(pdf_pages)

        self._save_chunks_to_store(chunks)

        return True, ''

    def _save_chunks_to_store(self, chunks: list[Document]) -> None:
        """Saves the processed document chunks to the vector store."""

        self._vector_store_proxy.store_documents(chunks)

    def _sanitize_pdf_metadata(self, pdf_pages: list[Document]) -> None:
        """Sanitizes metadata of PDF documents to remove unnecessary fields."""

        for page in pdf_pages:

            page.metadata = {
                'page': page.metadata.get('page_label', 'Unknown'),
                'title': page.metadata.get('title', 'Unknown'),
                'author': page.metadata.get('author', 'Unknown')
            }
