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
