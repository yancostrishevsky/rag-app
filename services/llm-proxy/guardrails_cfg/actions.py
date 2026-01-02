"""Contains custom actions triggered by the safety guardrails."""
# mypy: ignore-errors

import logging
from typing import Optional, Any

from langchain_core.language_models import BaseLLM

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.actions.llm.utils import LLMCallException


def _logger() -> logging.Logger:
    return logging.getLogger('llm_proxy._nemoguardrails.actions')


def _construct_conversation_fragment(context: dict[str, Any]) -> str:
    """Constructs a conversation fragment from the llm call context."""

    conversation = []

    if 'last_user_message' in context:
        conversation.append(f"User: {context['last_user_message']}")

    if 'last_bot_message' in context:
        conversation.append(f"Assistant: {context['last_bot_message']}")

    if 'user_message' in context:
        conversation.append(f"User: {context['user_message']}")

    return "\n".join(conversation)


@action(is_system_action=False)
async def chat_conversation_check(llm_task_manager: LLMTaskManager,
                                  context: Optional[dict] = None,
                                  llm: Optional[BaseLLM] = None,
                                  config: Optional[RailsConfig] = None
                                  ) -> dict[str, Any]:
    """Checks the user input together with the chat history to determine if it is safe to answer."""

    _MAX_TOKENS = 3

    conversation_fragment = _construct_conversation_fragment(context or {})

    if conversation_fragment:

        prompt = llm_task_manager.render_task_prompt(
            task='chat_conversation_check',
            context={
                "conversation_fragment": conversation_fragment,
            },
        )
        stop = llm_task_manager.get_stop_tokens(task='chat_conversation_check')
        max_tokens = llm_task_manager.get_max_tokens(task='chat_conversation_check')
        max_tokens = max_tokens or _MAX_TOKENS

        llm_call_info_var.set(LLMCallInfo(task='chat_conversation_check'))

        try:
            response = await llm_call(
                llm,
                prompt,
                stop=stop,
                llm_params={
                    "temperature": config.lowest_temperature,
                    "max_tokens": max_tokens
                },
            )

            _logger().debug("chat_conversation_check response is `%s` for prompt:\n%s",
                            response, prompt)

            result = llm_task_manager.parse_task_output('chat_conversation_check',
                                                        output=response,
                                                        forced_output_parser="is_content_safe")

            if result[0]:
                return {'result': True, 'reason': None}

            else:
                return {
                    'result': False,
                    'reason': '<input_rails_violation>'}

        except LLMCallException as e:
            _logger().error('LLM call failed during chat_conversation_check: %s', e)

            return {'result': False, 'reason': '<llm_call_error>'}
