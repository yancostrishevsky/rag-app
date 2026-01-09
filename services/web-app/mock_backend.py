"""Mock backend service for testing the web application."""
import asyncio
import json
import logging
import random
from typing import Any
from typing import AsyncIterator
from typing import Dict
from typing import List

import hydra
import omegaconf
import pydantic
import uvicorn
import fastapi
from fastapi.responses import StreamingResponse


# pylint: disable-all
# mypy: ignore-errors

app = fastapi.FastAPI()


class ConversationState(pydantic.BaseModel):
    """State of the chat conversation including chat history and current user query."""
    chat_history: list[dict[str, Any]]
    user_message: str


class RequestStreamChatResponse(pydantic.BaseModel):
    """Request to return the LLM response for a given query and retrieved context."""
    conversation_state: ConversationState
    context_docs: list[dict[str, Any]]


@app.post('/stream_chat_response')
async def stream_chat_response(request: RequestStreamChatResponse) -> StreamingResponse:
    """"""

    mock_response = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
    exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor
    in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur
    sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est
    laborum."""

    mock_words = mock_response.split()

    response = ' '.join(random.sample(mock_words, len(mock_words)))
    response += f'Documents used: {[doc["metadata"]["title"] for doc in request.context_docs]}'

    async def event_generator() -> AsyncIterator[bytes]:

        if 'fail' in request.conversation_state.user_message.lower():
            chunk = {'error': 'Simulated backend failure.'}
            yield json.dumps(chunk).encode('utf-8')
            return

        for token in response.replace(' ', ' [split_token]') .split('[split_token]'):
            chunk = {'content': token}
            yield json.dumps(chunk).encode('utf-8')
            await asyncio.sleep(0.05)

    return StreamingResponse(event_generator(), media_type='application/json')


class ResponseInputCheck(pydantic.BaseModel):

    is_ok: bool
    reason: str | None = None


@app.post('/check_input_safety')
async def check_input_safety(request: ConversationState) -> ResponseInputCheck:

    if 'badword' in request.user_message.lower():
        return ResponseInputCheck(is_ok=False, reason='Inappropriate language detected.')

    return ResponseInputCheck(is_ok=True)


@app.post('/check_input_relevance')
async def check_input_relevance(request: ConversationState) -> ResponseInputCheck:

    if 'unrelated' in request.user_message.lower():
        return ResponseInputCheck(
            is_ok=False, reason='Input is not relevant.')

    return ResponseInputCheck(is_ok=True)


class RequestCollectContextInfo(pydantic.BaseModel):
    user_message: str
    chat_history: list[dict[str, Any]]


class ResponseCollectContextInfo(pydantic.BaseModel):
    context_docs: list[dict[str, Any]]


@app.post('/collect_context_info')
async def collect_context_info(request: RequestCollectContextInfo) -> ResponseCollectContextInfo:

    await asyncio.sleep(3.0)

    logging.info('/collect_context_info - Message: %s', request.user_message)

    mock_docs = [
        {'content': 'This is document 1', 'metadata': {
            'title': 'Document 1',
            'author': 'Author A',
            'page': '1'}},
        {'content': 'This is document 2', 'metadata': {
            'title': 'Document 2',
            'author': 'Author B',
            'page': '2'}},
        {'content': 'This is document 3', 'metadata': {
            'title': 'Document 3',
            'author': 'Author C',
            'page': '3'}},
    ]

    return ResponseCollectContextInfo(context_docs=random.sample(mock_docs, 2))


class ResponseUploadDocument(pydantic.BaseModel):
    error: str | None = None


@app.post('/upload_pdf')
async def upload_pdf(file: fastapi.UploadFile) -> ResponseUploadDocument:

    if not file.filename.lower().endswith('.pdf'):
        return ResponseUploadDocument(error='Only PDF files are supported.')

    return ResponseUploadDocument()


@hydra.main(version_base=None, config_path='cfg', config_name='mock_backend')
def main(cfg: omegaconf.DictConfig) -> None:
    """Sets up the mock backend service."""

    logging.info('Starting mock backend with configuration:\n%s',
                 omegaconf.OmegaConf.to_yaml(cfg))

    uvicorn.run(
        'mock_backend:app',
        host=cfg.host,
        port=cfg.port,
        reload=False
    )


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
