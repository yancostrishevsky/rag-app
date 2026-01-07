""""Starts the backend server for the entrypoint service."""
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import fastapi
import hydra
import omegaconf
import pydantic
import uvicorn

from context_retriever import doc_preparation_service
from context_retriever import doc_retrieval_service
from context_retriever.vector_store import qdrant as qdrant_vs
from context_retriever.vector_store.core import EmbeddingModelCfg


def _logger() -> logging.Logger:
    return logging.getLogger('context_retriever')


doc_prep: doc_preparation_service.DocPreparationService | None = None  # pylint: disable=invalid-name
doc_retrieval: doc_retrieval_service.DocRetrievalService | None = None  # pylint: disable=invalid-name


@asynccontextmanager
async def lifespan(_):  # type: ignore
    """Sets up global contexts used by the server workers."""

    cfg_serialized = os.environ.get('CONTEXT_RETRIEVER_SERVER_CFG', None)

    if cfg_serialized is None:
        _logger().critical('Failed to load context_retriever server config from env.')
        sys.exit(1)

    cfg = omegaconf.OmegaConf.create(json.loads(cfg_serialized))

    # The llm_service is designed to be used by the endpoint handlers as a global service. It is
    # not assigned in the `main` function because the uvicorn workers don't call it, as opposed to
    # the `lifespan` callback.
    global doc_prep  # pylint: disable=global-statement
    global doc_retrieval  # pylint: disable=global-statement

    if cfg.vector_store.driver == 'qdrant':
        vector_store_proxy = qdrant_vs.QdrantProxy(
            qdrant_vs.QdrantProxyCfg(
                url=cfg.vector_store.url,
                collection_name=cfg.vector_store.collection_name,
                uploading_batch_size=cfg.vector_store.uploading_batch_size,
                embedding_model=EmbeddingModelCfg(
                    **cfg.vector_store.embedding_model
                )
            )
        )
    else:
        _logger().critical('Unsupported vector store driver: %s', cfg.vector_store.driver)
        sys.exit(1)

    doc_prep = doc_preparation_service.DocPreparationService(
        vector_store_proxy=vector_store_proxy,
        doc_processing_cfg=doc_preparation_service.DocProcessingCfg(
            **cfg.doc_processing)
    )
    doc_retrieval = doc_retrieval_service.DocRetrievalService()

    yield


app = fastapi.FastAPI(lifespan=lifespan)


@app.get('/ping')
async def read_ping() -> dict[str, str]:
    """Health check endpoint."""
    return {'message': 'Service is running'}


class RequestCollectContextInfo(pydantic.BaseModel):
    """Request to collect context documents from the vector store."""
    user_message: str
    chat_history: list[dict[str, Any]]


class ResponseCollectContextInfo(pydantic.BaseModel):
    """Response from context-retriever after collecting context documents."""
    context_docs: list[dict[str, Any]]


@app.post('/collect_context_info')
async def collect_context_info(request: RequestCollectContextInfo) -> ResponseCollectContextInfo:
    """Collects context documents from the vector store based on user message and chat history."""
    return ResponseCollectContextInfo(context_docs=[])


class ResponseUploadDocument(pydantic.BaseModel):
    """Response from context-retriever to web-app after uploading a document."""
    error: str | None = None


@app.post('/upload_pdf')
async def upload_pdf(file: fastapi.UploadFile) -> ResponseUploadDocument:
    """Processes and saves an uploaded PDF document to the vector store."""

    assert doc_prep is not None

    if file.content_type is None:
        return ResponseUploadDocument(error='Cannot determine content type for the uploaded PDF.')

    if file.size is None:
        return ResponseUploadDocument(error='Cannot determine file size for the uploaded PDF.')

    is_ok, error_msg = doc_prep.upload_pdf(
        file_size=file.size,
        content_type=file.content_type,
        file_stream=file.file
    )

    if not is_ok:
        return ResponseUploadDocument(error=error_msg)

    return ResponseUploadDocument()


def _configure_logging(script_cfg: omegaconf.DictConfig) -> None:

    logging.config.dictConfig({
        'version': 1,
        'loggers': {
            'root': {
                'level': 'WARNING',
                'handlers': ['console', 'file']
            },
            'context_retriever': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'default'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'default',
                'filename': os.path.join(script_cfg.persist_data_path,
                                         'log',
                                         f'{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log'),
                'maxBytes': 5_000_000,
                'backupCount': 5,
                'encoding': 'utf-8',
            }
        },
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            }
        }
    })


@hydra.main(version_base=None, config_path='cfg', config_name='main')
def main(cfg: omegaconf.DictConfig) -> None:
    """Initializes and serves the web app."""

    _logger().info('Script configuration:\n%s', omegaconf.OmegaConf.to_yaml(cfg))

    os.makedirs(os.path.join(cfg.persist_data_path, 'log'), exist_ok=True)

    _configure_logging(cfg)

    _logger().info('Starting server on %s:%d with %d workers.',
                   cfg.server_host, cfg.server_port, cfg.n_server_workers)

    cfg_serialized = json.dumps(omegaconf.OmegaConf.to_container(cfg))
    os.environ['CONTEXT_RETRIEVER_SERVER_CFG'] = cfg_serialized

    uvicorn.run('main:app',
                host=cfg.server_host,
                port=cfg.server_port,
                workers=cfg.n_server_workers,
                log_level='info',
                access_log=True,
                limit_concurrency=1000,
                timeout_keep_alive=5)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
