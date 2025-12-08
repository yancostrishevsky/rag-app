"""Mock backend service for testing the web application."""
import asyncio
import json
import logging
import random
from typing import AsyncIterator
from typing import Dict
from typing import List
from typing import Tuple

import hydra
import omegaconf
import pydantic
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


app = FastAPI()


class RequestCollectContextInfo(pydantic.BaseModel):
    """Request coming from the web app to collect context information."""
    user_message: str
    chat_history: List[Dict[str, str]]


class ResponseCollectContextInfo(pydantic.BaseModel):
    """Response sent back to the web app with collected context information."""
    context_docs: List[Tuple[str, str]]


class RequestStreamChatResponse(pydantic.BaseModel):
    """Request coming from the web app after collecting the context to stream chat responses."""
    user_message: str
    chat_history: List[Dict[str, str]]
    context_docs: List[Tuple[str, str]]


@app.get('/ping')
async def read_ping() -> Dict[str, str]:
    """Health check endpoint."""
    return {'message': 'Service is running'}


@app.post('/collect_context_info')
async def collect_context_info(request: RequestCollectContextInfo) -> ResponseCollectContextInfo:
    """Collects context information from database for a given query."""

    logging.info('/collect_context_info - Message: %s', request.user_message)

    mock_docs = [
        ('doc1', 'This is the content of document 1.'),
        ('doc2', 'This is the content of document 2.'),
        ('doc3', 'This is the content of document 3.'),
    ]

    return ResponseCollectContextInfo(context_docs=mock_docs)


@app.post('/stream_chat_response')
async def stream_chat_response(request: RequestStreamChatResponse) -> StreamingResponse:
    """Streams chat response for the user query based on the provided context."""

    mock_response = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
    exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor
    in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur
    sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est
    laborum."""

    mock_words = mock_response.split()

    response = ' '.join(random.sample(mock_words, len(mock_words)))
    response += f'Documents used: {[doc_id for doc_id, _ in request.context_docs]}'

    async def event_generator() -> AsyncIterator[bytes]:
        for token in response.replace(' ', ' [split_token]') .split('[split_token]'):
            chunk = {'content': token}
            yield json.dumps(chunk).encode('utf-8')
            await asyncio.sleep(0.05)

    return StreamingResponse(event_generator(), media_type='application/json')


@hydra.main(version_base=None, config_path='cfg', config_name='mock_backend')
def main(cfg: omegaconf.DictConfig) -> None:
    """Sets up the mock backend service."""

    logging.info('Starting mock backend with configuration: %s',
                 omegaconf.OmegaConf.to_yaml(cfg))

    uvicorn.run(
        'mock_backend:app',
        host=cfg.host,
        port=cfg.port,
        reload=False
    )


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
