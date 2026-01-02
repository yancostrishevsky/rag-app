"""Contains service responsible for connecting with self-hosted LLM."""
import json
import logging
from typing import Any
from typing import AsyncIterator
from typing import Dict
from typing import List

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

        async for chunk in self._rails_client.stream_async(
                messages=messages,
                options={'log': {'activated_rails': True},
                         'llm_output': True}):

            if self._is_chunk_error(chunk):
                yield json.dumps({'error': self._get_error_message(chunk)}).encode('utf-8')
                return

            chunk_struct = {'content': chunk}

            yield json.dumps(chunk_struct).encode('utf-8')

    def _is_chunk_error(self, chunk: str) -> bool:
        """Tells whether the given chunk indicates an error.

        An error chunk does not contain the expected vocabulary and indicates that either
        the guardrails blocked the response or the LLM call failed.
        """

        return chunk in ('<input_rails_violation>', '<llm_call_error>')

    def _get_error_message(self, chunk: str) -> str:
        """Returns human-readable error message for the given error chunk."""

        if chunk == '<input_rails_violation>':
            return 'The input was blocked by safety guardrails.'

        if chunk == '<llm_call_error>':
            return 'The model call failed.'

        _logger().error('Unknown error chunk: %s', chunk)
        return 'An unknown error occurred.'
