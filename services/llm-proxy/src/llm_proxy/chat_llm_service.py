"""Contains service responsible for connecting with self-hosted LLM."""

import logging
from typing import Dict
from typing import List
from typing import Any
from typing import AsyncIterator
import json

import nemoguardrails


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ChatLLMService:
    """Establishes connection with self-hosted LLM and handles requests to it."""

    def __init__(self,
                 guardrails_cfg_path: str):

        config = nemoguardrails.RailsConfig.from_path(guardrails_cfg_path)

        self._rails_client = nemoguardrails.LLMRails(config)

        _logger().debug('Created GuardrailsService from config: %s.', config)

    async def stream_chat_response(self,
                                   user_query: str,
                                   chat_history: List[Dict[str, Any]]) -> AsyncIterator[bytes]:
        """Streams, chunk by chunk, LLM response for a given query and chat history.

        The input is checkes according to the guardrails specification.
        """

        _logger().debug('Streaming llm response for query \'%s\' and conversation %s...',
                        user_query, chat_history)

        messages = chat_history + [
            {'role': 'user', 'content': user_query}
        ]

        async for chunk in self._rails_client.stream_async(messages=messages):

            chunk_struct = {'content': chunk}
            yield json.dumps(chunk_struct).encode('utf-8')
