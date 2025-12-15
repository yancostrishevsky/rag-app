""""Starts the backend server for the entrypoint service."""
import logging
import multiprocessing
import os
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from contextlib import asynccontextmanager


import hydra
import omegaconf
import uvicorn
import pydantic
import fastapi
from fastapi.responses import StreamingResponse

from llm_proxy import (guardrails_service, chat_llm_service)


def _logger() -> logging.Logger:
    return logging.getLogger()


guard_service: guardrails_service.GuardrailsService | None = None  # pylint: disable=invalid-name
llm_service: chat_llm_service.ChatLLMService | None = None  # pylint: disable=invalid-name


@asynccontextmanager
async def lifespan(_):  # type: ignore
    """Sets up global contexts used by the server workers."""

    cfg = omegaconf.OmegaConf.load('cfg/main.yaml')

    global guard_service, llm_service  # pylint: disable=global-statement

    guard_service = guardrails_service.GuardrailsService(
        config_path=cfg.guardrails_cfg_path
    )
    llm_service = chat_llm_service.ChatLLMService(
        ollama_model=cfg.chat_llm_cfg.model,
        ollama_url=cfg.chat_llm_cfg.url
    )

    yield


app = fastapi.FastAPI(lifespan=lifespan)


@app.get('/ping')
async def read_ping() -> Dict[str, str]:
    """Health check endpoint."""
    return {'message': 'Service is running'}


class RequestStreamChatResponse(pydantic.BaseModel):
    """Request to return the LLM response for a given query and retrieved context."""
    user_message: str
    chat_history: List[Dict[str, Any]]
    context_docs: List[Dict[str, Any]]


@app.post('/stream_chat_response')
async def stream_chat_response(request: RequestStreamChatResponse) -> StreamingResponse:
    """Streams chat response for the user query based on the provided context."""

    assert llm_service is not None

    return StreamingResponse(llm_service.stream_chat_response(request.user_message,
                                                              chat_history=request.chat_history),
                             media_type='application/json')


class RequestShouldAllowQuery(pydantic.BaseModel):
    """Request to tell whether a given query is eligible for processing."""
    user_message: str
    chat_history: List[Dict[str, Any]]


class ResponseShouldAllowQuery(pydantic.BaseModel):
    """Responds with the information whether a given query should be processed."""
    allowed: bool


@app.post('/should_allow_query')
async def should_allow_query(request: RequestShouldAllowQuery) -> ResponseShouldAllowQuery:
    """Tells whether a given user query passes the guardrails."""

    assert guard_service is not None

    is_allowed = await guard_service.should_process_query(query=request.user_message,
                                                          chat_history=request.chat_history)

    return ResponseShouldAllowQuery(allowed=is_allowed)


@hydra.main(version_base=None, config_path='cfg', config_name='main')
def main(cfg: omegaconf.DictConfig) -> None:
    """Initializes and serves the web app."""

    os.makedirs(os.path.join(cfg.persist_data_path, 'log'), exist_ok=True)

    logging.config.dictConfig({
        'version': 1,
        'loggers': {
            'root': {
                'level': 'NOTSET',
                'handlers': ['console', 'file'],
                'propagate': True
            },
            'httpx': {
                'level': 'WARNING',
                'propagate': False
            },
            'httpcore': {
                'level': 'WARNING',
                'propagate': False
            },
            'nemoguardrails': {
                'level': 'WARNING',
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
                'filename': os.path.join(cfg.persist_data_path,
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

    # n_workers = (multiprocessing.cpu_count() * 2) + 1
    n_workers = 1

    _logger().info('Starting server on %s:%d with %d workers.',
                   cfg.server_host, cfg.server_port, n_workers)

    uvicorn.run('main:app',
                host=cfg.server_host,
                port=cfg.server_port,
                workers=n_workers,
                log_level='info',
                access_log=True,
                limit_concurrency=1000,
                timeout_keep_alive=5)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
