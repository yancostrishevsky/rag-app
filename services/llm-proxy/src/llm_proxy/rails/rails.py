"""Contains implementation of input/output/retrieval guardrails."""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from llm_proxy.rails.core import Guardrail, LLMCallContext, GuardrailDecision


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ConversationSafetyGuardrail(Guardrail):
    """Guardrail that checks whether the conversation is safe.

    The guardrail checks for harmful or inappropriate content in the user messages. according
    to the MLCommons AI Ethics Guidelines.
    """

    _SYSTEM_PROMPT = """You are a professional content moderator. Your task is to determine
    whether the content contains unsafe or inappropriate material according to the provided
    guidelines."""

    _MAIN_PROMPT_TEMPLATE = """
        Your task is to determine whether the given user query is safe and appropriate, according
        to the following guidelines:

        <BEGIN_UNSAFE_CONTENT_CATEGORIES>
        S1. Hate Speech.

        <END_UNSAFE_CONTENT_CATEGORIES>
        Pay heed to the entire conversation history when making your decision.

        <BEGIN_CONVERSATION>
        {conversation}

        <END_CONVERSATION>
        Your response should contain two lines:
        1. One word: 'safe' if the conversation is safe, or 'unsafe' if it is not.
        2. A brief explanation of your decision based on the guidelines.
    """

    def __init__(self, llm: BaseChatModel) -> None:

        self._llm = llm

    async def should_pass(self, llm_call_context: LLMCallContext) -> GuardrailDecision:
        """Returns true if the conversation is safe, false otherwise."""

        messages = [SystemMessage(self._SYSTEM_PROMPT),
                    HumanMessage(self._MAIN_PROMPT_TEMPLATE.format(
                        conversation=self._format_conversation(llm_call_context)))]

        response = await self._llm.ainvoke(messages)

        decision = str(response.content)
        decision_lines = [line.strip() for line in decision.split('\n') if line.strip()]

        _logger().debug('Guardrail \'%s\' response:\n%s', self.name, decision)

        if len(decision_lines) != 2:
            return GuardrailDecision(
                should_pass=False,
                reason='Guardrail execution failed.'
            )

        if decision_lines[0].lower() == 'safe':
            return GuardrailDecision(should_pass=True)

        return GuardrailDecision(
            should_pass=False,
            reason=decision_lines[1]
        )

    @property
    def name(self) -> str:
        return "ConversationSafetyGuardrail"

    def _format_conversation(self,
                             llm_call_context: LLMCallContext) -> str:
        """Formats the conversation history into a string."""

        conversation_lines = []

        for message in llm_call_context.chat_history + [
            {'role': 'user', 'content': llm_call_context.user_message}
        ]:
            role = message['role']
            content = message['content']

            if role == 'user':
                conversation_lines.append(f'User: {content}')
            else:
                conversation_lines.append(f'Assistant: {content}')

        return '\n'.join(conversation_lines)
