"""Contains functions to communicate with the entrypoint backend service."""
import json
import logging
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple
from typing import TypeAlias

import httpx
import requests

ChatHistoryType: TypeAlias = List[Dict[str, str]]


def _logger() -> logging.Logger:
    return logging.getLogger('web_app')


def _sanitize_chat_history(chat_history: ChatHistoryType) -> ChatHistoryType:
    """Sanitizes the chat history to ensure it contains only relevant fields."""
    return [
        {
            'role': item['role'],
            'content': item['content']
        }
        for item in chat_history
    ]


class BackendService:
    """Communicates with the entrypoint-backend."""

    def __init__(self, backend_url: str):

        self._backend_url = backend_url

    def collect_context_info(self,
                             user_message: str,
                             chat_history: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Collects context information based on the user's message and chat history.

        Args:
            user_message: The message from the user to collect context information for.
            chat_history: The history of the chat to provide context for the request.

        Raises:
            requests.HTTPError: If the request to the backend fails.

        Returns:
            A list of tuples where each tuple contains a document title and its content.
        """

        _logger().debug('Collecting context info with user_message: %s and chat_history: %s',
                        user_message, chat_history)

        chat_history = _sanitize_chat_history(chat_history)

        url = f"{self._backend_url}/collect_context_info"
        payload = {
            'user_message': user_message,
            'chat_history': chat_history
        }

        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()

        response_data = response.json()

        return list((title, content) for title, content in response_data['context_docs'])

    def stream_chat_response(self,
                             user_message: str,
                             chat_history: ChatHistoryType,
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

        chat_history = _sanitize_chat_history(chat_history)

        url = f"{self._backend_url}/stream_chat_response"

        payload = {
            'user_message': user_message,
            'chat_history': chat_history,
            'context_docs': context_docs
        }

        with httpx.stream('POST', url, json=payload, timeout=5) as stream:
            for chunk in stream.iter_bytes():
                yield json.loads(chunk.decode('utf-8'))
