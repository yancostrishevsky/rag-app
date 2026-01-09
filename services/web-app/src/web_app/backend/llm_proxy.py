"""Contains service that communicates with the llm-proxy module."""
import json
import logging
from typing import Iterator
import requests

import httpx
from web_app.backend import utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class LLMProxyService:
    """Communicates with the llm-proxy module and returns model's responses."""

    def __init__(self, endpoint_cfg: utils.EndpointConnectionCfg):

        self._endpoint_cfg = endpoint_cfg

        _logger().info('Created service for llm-proxy with cfg: %s',
                       endpoint_cfg)

    def stream_chat_response(self,
                             user_message: str,
                             chat_history: utils.ChatHistory,
                             context_docs: list[utils.ContextDocument]) -> Iterator[dict[str, str]]:
        """Collects LLM response based on the context and streams it.

        Args:
            user_message: The message from the user to generate a response for.
            chat_history: The history of the chat to provide context for the request.
            context_docs: The documents retrieved to provide additional context.

        Returns:
            A generator that yields chunks of the chat response as they are received. Each chunk
                contain the 'content' field.
        """

        _logger().debug(('Streaming chat response with user_message: %s, ' +
                         'chat_history: %s, context_docs: %s'),
                        user_message, chat_history, context_docs)

        url = f"{self._endpoint_cfg.url}/stream_chat_response"

        payload = {
            'conversation_state': {
                'user_message': user_message,
                'chat_history': utils.chat_history_to_payload(chat_history)
            },
            'context_docs': utils.context_docs_to_payload(context_docs)
        }

        with httpx.stream('POST', url,
                          json=payload,
                          timeout=self._endpoint_cfg.connection_timeout) as stream:
            for chunk in stream.iter_bytes():
                yield json.loads(chunk.decode('utf-8'))

    def check_input_safety(self,
                           user_message: str,
                           chat_history: utils.ChatHistory,
                           ) -> utils.InputCheckResult:
        """Sends the conversation state to the llm-proxy to check input safety.

        Raises:
            requests.HTTPError: If the request to the llm-proxy fails.
        """

        _logger().debug(('Checking input safety with user_message: %s, chat_history: %s'),
                        user_message, chat_history)

        return self._sanitize_input(
            user_message=user_message,
            chat_history=chat_history,
            url=f"{self._endpoint_cfg.url}/check_input_safety"
        )

    def check_input_relevance(self,
                              user_message: str,
                              chat_history: utils.ChatHistory,
                              ) -> utils.InputCheckResult:
        """Sends the conversation state to the llm-proxy to check input relevance.

        Raises:
              requests.HTTPError: If the request to the llm-proxy fails.
        """

        _logger().debug(('Checking input relevance with user_message: %s, chat_history: %s'),
                        user_message, chat_history)

        return self._sanitize_input(
            user_message=user_message,
            chat_history=chat_history,
            url=f"{self._endpoint_cfg.url}/check_input_relevance"
        )

    def _sanitize_input(self,
                        user_message: str,
                        chat_history: utils.ChatHistory,
                        url: str,
                        ) -> utils.InputCheckResult:
        """Sends the conversations state to a chosen llm-proxy guardrail endpoints."""

        payload = {
            'user_message': user_message,
            'chat_history': utils.chat_history_to_payload(chat_history)
        }

        try:
            response = requests.post(url,
                                     json=payload,
                                     timeout=self._endpoint_cfg.connection_timeout)

        except requests.exceptions.ConnectionError as e:
            _logger().error('Connection error while connecting to llm-proxy: %s', e)
            raise requests.HTTPError('Connection error while connecting to llm-proxy.') from e

        response.raise_for_status()

        resp_json = response.json()
        return utils.InputCheckResult(is_ok=resp_json['is_ok'], reason=resp_json['reason'])
