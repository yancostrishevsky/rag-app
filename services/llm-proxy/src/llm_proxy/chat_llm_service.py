"""Contains service responsible for connecting with self-hosted LLM."""
import json
import logging
from typing import Any
from typing import AsyncIterator

from langchain_ollama import ChatOllama
from llm_proxy import llm_actions
from llm_proxy.rails import rails
from llm_proxy.rails.core import Guardrail
from llm_proxy.rails.core import LLMCallContext


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ChatLLMService:
    """Establishes connection with self-hosted LLM and handles requests to it."""

    def __init__(self, models_cfg: dict[str, dict[str, Any]]) -> None:

        self._input_guardrails: list[Guardrail] = [
            rails.ConversationSafetyGuardrail(
                llm=ChatOllama(**models_cfg['conversation_safety_guardrail'],
                               temperature=0.0))
        ]

        self._chat_response_action = llm_actions.ChatResponseAction(
            llm=ChatOllama(**models_cfg['main_chat'])
        )

    async def stream_chat_response(self,
                                   user_query: str,
                                   chat_history: list[dict[str, Any]],
                                   context_documents: list[dict[str, Any]] | None = None,
                                   ) -> AsyncIterator[bytes]:
        """Streams, chunk by chunk, LLM response for a given query and chat history.

        The input is checkes according to the guardrails specification.
        """

        _logger().debug('Streaming llm response for query \'%s\' and conversation %s...',
                        user_query, chat_history)

        llm_call_context = LLMCallContext(
            user_message=user_query,
            chat_history=chat_history,
            retrieved_context=[]
        )

        for guardrail in self._input_guardrails:
            decision = await guardrail.should_pass(llm_call_context)

            if not decision.should_pass:
                error_message = (
                    f'Input guardrail \'{guardrail.name}\' blocked the request. '
                    f'Reason: {decision.reason}'
                )

                _logger().warning(error_message)

                yield json.dumps({'error': error_message}).encode('utf-8')
                return

        try:
            async for chunk in self._chat_response_action.run(
                    user_query=user_query,
                    chat_history=chat_history,
                    context_documents=context_documents
            ):
                chunk_struct = {'content': chunk}

                yield json.dumps(chunk_struct).encode('utf-8')

        except Exception as e:  # pylint: disable=broad-except
            _logger().error('Chat call failed: %s', str(e))

            yield json.dumps({'error': 'Internal system error.'}).encode('utf-8')
