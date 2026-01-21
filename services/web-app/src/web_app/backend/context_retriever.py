"""Contains implementation of service that communicates with the context retriever module."""
import logging

import magic
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

    def upload_file(self,
                    uploaded_file_path: str) -> str | None:
        """Sends a file to the context retriever service for processing.

        Raises:
            requests.HTTPError: If the request to the backend fails.

        Returns:
            None, if the upload was successful, or an error message string if there was an error.
        """

        _logger().debug('Uploading file %s to context retriever service.', uploaded_file_path)

        mime = magic.from_file(uploaded_file_path, mime=True)

        if mime == 'application/pdf':
            url = f"{self._endpoint_cfg.url}/upload_pdf"

        else:
            return 'Only PDF files are supported.'

        try:
            with open(uploaded_file_path, 'rb') as file:
                response = requests.post(
                    url,
                    files={'file': (uploaded_file_path, file, mime)},
                    timeout=self._endpoint_cfg.connection_timeout
                )

        except requests.exceptions.ConnectionError as e:
            _logger().error('Connection error while uploading PDF file to context retriever: %s', e)
            raise requests.HTTPError(
                'Connection error while uploading PDF file to context retriever.') from e

        response.raise_for_status()

        response_data = response.json()

        if response_data['error'] is not None:
            return str(response_data['error'])

        return None
