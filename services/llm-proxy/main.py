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

from llm_proxy import chat_llm_service


def _logger() -> logging.Logger:
    return logging.getLogger()


llm_service: chat_llm_service.ChatLLMService | None = None  # pylint: disable=invalid-name


@asynccontextmanager
async def lifespan(_):  # type: ignore
    """Sets up global contexts used by the server workers."""

    cfg = omegaconf.OmegaConf.load('cfg/main.yaml')

    global llm_service  # pylint: disable=global-statement

    llm_service = chat_llm_service.ChatLLMService(
        guardrails_cfg_path=cfg.guardrails_cfg_path
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

    _logger().info('Starting server on %s:%d with %d workers.',
                   cfg.server_host, cfg.server_port, cfg.n_server_workers)

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
