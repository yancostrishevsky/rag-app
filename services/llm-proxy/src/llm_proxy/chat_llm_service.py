"""Contains service responsible for connecting with self-hosted LLM."""

import logging
from typing import Dict
from typing import List
from typing import Any
from typing import AsyncIterator
import json

import ollama


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ChatLLMService:
    """Establishes connection with self-hosted LLM and handles requests to it."""

    def __init__(self,
                 ollama_model: str,
                 ollama_url: str):

        self._ollama_model = ollama_model
        self._ollama_url = ollama_url

        self._client = ollama.AsyncClient(ollama_url)

        _logger().debug('Created ChatLLMService for model: %s and url: %s.',
                        ollama_model, ollama_url)

    async def stream_chat_response(self,
                                   user_query: str,
                                   chat_history: List[Dict[str, Any]]) -> AsyncIterator[bytes]:
        """Streams, chunk by chunk, LLM response for a given query and chat history."""

        messages = chat_history + [
            {'role': 'user', 'content': user_query}
        ]

        async for chunk in await self._client.chat(model=self._ollama_model,
                                                   messages=messages,
                                                   stream=True):

            chunk_struct = {'content': chunk}
            yield json.dumps(chunk_struct).encode('utf-8')
