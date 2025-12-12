"""Contains service that communicates with the llm-proxy module."""

from typing import Tuple
from typing import Dict
from typing import List
from typing import Iterator
import logging
import httpx
import json

from web_app.backend import utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class LLMProxyService:
    """Communicates with the llm-proxy module and returns model's responses."""

    def __init__(self, llm_proxy_url: str):

        self._llm_proxy_url = llm_proxy_url

    def stream_chat_response(self,
                             user_message: str,
                             chat_history: utils.ChatHistoryType,
                             context_docs: List[Tuple[str, str]]) -> Iterator[Dict[str, str]]:
        """Collects LLM response based on the context and streams it.

        Args:
            user_message: The message from the user to generate a response for.
            chat_history: The history of the chat to provide context for the request.
            context_docs: The documents retrieved to provide additional context.

        Returns:
            A generator that yields chunks of the chat response as they are received.
        """

        _logger().debug(('Streaming chat response with user_message: %s, ' +
                         'chat_history: %s, context_docs: %s'),
                        user_message, chat_history, context_docs)

        chat_history = utils.sanitize_chat_history(chat_history)

        url = f"{self._llm_proxy_url}/stream_chat_response"

        payload = {
            'user_message': user_message,
            'chat_history': chat_history,
            'context_docs': context_docs
        }

        with httpx.stream('POST', url, json=payload, timeout=5) as stream:
            for chunk in stream.iter_bytes():
                yield json.loads(chunk.decode('utf-8'))
