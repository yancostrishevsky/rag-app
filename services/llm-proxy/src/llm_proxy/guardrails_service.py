"""Contains the implementation of a guardrails service performing query sanitization."""

from typing import Dict
from typing import Any
from typing import List
import logging

import nemoguardrails


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class GuardrailsService:
    """Checks whether LLM inputs and outputs should be allowed to pass.

    The queries and responses are checked according to a given configuration that specifies llm
    endpoints and guardrails.
    """

    def __init__(self,
                 config_path: str):

        config = nemoguardrails.RailsConfig.from_path(config_path)

        self._guardrails_client = nemoguardrails.LLMRails(config)

        _logger().debug('Created GuardrailsService from config path: %s.', config_path)

    async def should_process_query(self,
                                   query: str,
                                   chat_history: List[Dict[str, Any]]) -> bool:
        """Tells whether a user's chat message (query + chat history) passes the guardrails."""

        _logger().debug('Matching the query \'%s\' and conversation %s against input guardrails...',
                        query, chat_history)

        llm_result = await self._guardrails_client.generate_async(
            messages=chat_history + [{"role": "user", "content": query}],
            options={"rails": ["input"], },
        )

        _logger().debug('Received response: %s', llm_result)

        return llm_result.response[0]['content'].strip().lower() == 'yes'
