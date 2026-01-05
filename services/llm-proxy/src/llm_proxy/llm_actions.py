"""Contains modules performing LLM-related isolated actions."""
from typing import Any
from typing import AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage


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
                  context_documents: dict[str, Any] | None = None) -> AsyncIterator[str]:
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
            context_content = '\n\n'.join(context_documents)
            context_message = (
                'The following context documents are provided to help you answer the user query:\n'
                f"{context_content}"
            )
            messages.append(SystemMessage(content=context_message))

        messages.append(HumanMessage(content=user_query))

        async for chunk in self._llm.astream(messages):
            yield str(chunk.content)
