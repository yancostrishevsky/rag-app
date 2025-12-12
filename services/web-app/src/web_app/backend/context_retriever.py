"""Contains implementation of service that communicates with the context retriever module."""

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import requests

from web_app.backend import utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ContextRetrieverService:
    """Communicates with the context-retriever module and retrieves context for queries."""

    def __init__(self, context_retriever_url: str):

        self._context_retriever_url = context_retriever_url

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

        chat_history = utils.sanitize_chat_history(chat_history)

        url = f"{self._context_retriever_url}/collect_context_info"
        payload = {
            'user_message': user_message,
            'chat_history': chat_history
        }

        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()

        response_data = response.json()

        return list((title, content) for title, content in response_data['context_docs'])
