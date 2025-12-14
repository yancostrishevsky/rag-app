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

        _logger().debug('Created service for context-retriever with url: %s',
                        context_retriever_url)

    def collect_context_info(self,
                             user_message: str,
                             chat_history: utils.ChatHistory) -> List[utils.ContextDocument]:
        """Collects context information based on the user's message and chat history.

        Args:
            user_message: The message from the user to collect context information for.
            chat_history: The history of the chat to provide context for the request.

        Raises:
            requests.HTTPError: If the request to the backend fails.
        """

        _logger().debug('Collecting context info with user_message: %s and chat_history: %s',
                        user_message, chat_history)

        url = f"{self._context_retriever_url}/collect_context_info"
        payload = {
            'user_message': user_message,
            'chat_history': utils.chat_history_to_payload(chat_history)
        }

        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()

        response_data = response.json()

        return [utils.ContextDocument(doc['title'], doc['content'])
                for doc in response_data['context_docs']]
