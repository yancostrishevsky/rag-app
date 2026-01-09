"""Contains modules performing LLM-related isolated actions."""
from typing import Any
from typing import AsyncIterator
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ChatResponseAction:
    """Generates chat response for a given user query and chat history."""

    _SYSTEM_PROMPT = (
        'You are a helpful AI assistant. Respond to the user query based on the conversation '
        'history, i.e. fall back to the chat history when the query refers to previous messages.'
        'If any context documents are provided, use them to ground your response.'
    )

    def __init__(self,
                 llm: BaseChatModel):

        self._llm = llm

    async def run(self,
                  user_query: str,
                  chat_history: list[dict[str, str]],
                  context_documents: list[dict[str, Any]] | None = None) -> AsyncIterator[str]:
        """Generates chat response for the given user query and chat history."""

        messages: list[BaseMessage] = [SystemMessage(content=self._SYSTEM_PROMPT)]

        for message in chat_history:
            role = message['role']
            content = message['content']

            if role == 'user':
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        if context_documents:
            context_message = (
                'The following context documents are provided to help you answer the user query:\n'
                f"{self._format_context_documents(context_documents)}"
            )
            messages.append(SystemMessage(content=context_message))

        messages.append(HumanMessage(content=user_query))

        async for chunk in self._llm.astream(messages):
            yield str(chunk.content)

    def _format_context_documents(self,
                                  context_documents: list[dict[str, Any]]) -> str:
        """Formats context documents into a string for inclusion in the system prompt."""

        formatted_docs = []

        for doc in context_documents:
            formatted_doc = f"Title: {doc['metadata']['title']}\nContent: {doc['content']}"
            formatted_docs.append(formatted_doc)

        return '\n\n'.join(formatted_docs)


class SimpleConversationValidateAction:
    """Validates a conversation using a two line response format.

    The response is expected to contain:
    1. A single word indicating whether the input is 'good' or 'bad'.
    2. A brief explanation in words of the decision.
    """

    def __init__(self,
                 system_prompt: str,
                 main_prompt_template: str,
                 good_keyword: str,
                 llm: BaseChatModel
                 ) -> None:
        """Args:
            system_prompt: The system prompt to set the behavior of the LLM.
            main_prompt_template: The main prompt template containing a placeholder
                for the 'conversation' and 'user_input'.
            good_keyword: The keyword indicating a 'good' input in the LLM response.
                E.g. 'safe'.
            llm: The language model to use for validation.
        """

        self._system_prompt = system_prompt
        self._main_prompt_template = main_prompt_template
        self._good_keyword = good_keyword
        self._llm = llm

    async def run(self,
                  user_query: str,
                  chat_history: list[dict[str, str]]) -> tuple[bool, str | None]:
        """Returns a tuple indicating whether the input is safe and an optional reason."""

        messages = [SystemMessage(self._system_prompt),
                    HumanMessage(self._main_prompt_template.format(
                        conversation_history=self._format_conversation(chat_history),
                        user_input=user_query))]

        try:
            response = await self._llm.ainvoke(messages)

        except Exception as e:  # pylint: disable=broad-except
            _logger().error('Validation LLM call failed: %s', str(e))
            return False, 'Internal system error during conversation validation.'

        decision = str(response.content)
        decision_lines = [line.strip() for line in decision.split('\n') if line.strip()]

        _logger().debug('Validation LLM raw response:\n%s', decision)

        if len(decision_lines) != 2:
            _logger().error('Unexpected number of lines in validation response: %d',
                           len(decision_lines))
            return False, 'Internal system error during conversation validation.'

        if decision_lines[0].lower() == self._good_keyword.lower():
            return True, None

        return False, decision_lines[1]

    def _format_conversation(self,
                             chat_history: list[dict[str, str]]) -> str:
        """Formats the conversation history into a string."""

        conversation_lines = []

        for message in chat_history:
            role = message['role']
            content = message['content']

            if role == 'user':
                conversation_lines.append(f'User: {content}')
            else:
                conversation_lines.append(f'Assistant: {content}')

        return '\n'.join(conversation_lines)
