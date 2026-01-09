"""Contains implementation of service that communicates with the context retriever module."""
import logging

import requests
from web_app.backend import utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ContextRetrieverService:
    """Communicates with the context-retriever module and retrieves context for queries."""

    def __init__(self, endpoint_cfg: utils.EndpointConnectionCfg):

        self._endpoint_cfg = endpoint_cfg

        _logger().info('Created service for context-retriever with cfg: %s',
                       endpoint_cfg)

    def collect_context_info(self,
                             user_message: str,
                             chat_history: utils.ChatHistory) -> list[utils.ContextDocument]:
        """Collects context information based on the user's message and chat history.

        Args:
            user_message: The message from the user to collect context information for.
            chat_history: The history of the chat to provide context for the request.

        Raises:
            requests.HTTPError: If the request to the backend fails.
        """

        _logger().debug('Collecting context info with user_message: %s and chat_history: %s',
                        user_message, chat_history)

        url = f"{self._endpoint_cfg.url}/collect_context_info"
        payload = {
            'user_message': user_message,
            'chat_history': utils.chat_history_to_payload(chat_history)
        }

        response = requests.post(url, json=payload, timeout=self._endpoint_cfg.connection_timeout)
        response.raise_for_status()

        response_data = response.json()

        return [utils.ContextDocument(doc['content'], doc['metadata'])
                for doc in response_data['context_docs']]
