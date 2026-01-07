"""Contains service handling conversation context document retrieval."""

import logging
from typing import Any

import pydantic
from langchain_ollama.chat_models import ChatOllama
from langchain.messages import SystemMessage, HumanMessage

from context_retriever.vector_store import core as vs_core


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class HelperLLMConfig(pydantic.BaseModel):
    """Configuration for LLM model endpoint."""
    model_name: str
    url: str
    temperature: float


class DocRetrievalCfg(pydantic.BaseModel):
    """Configuration for document retrieval."""
    max_context_docs: int
    similarity_threshold: float
    helper_llm: HelperLLMConfig


class DocRetrievalService:
    """Retrieves context documents from the vector store.

    The service constructs queries based on user messages and chat history to fetch
    relevant documents.
    """

    _CONSTRUCT_QUERY_SYSTEM_PROMPT = """
    You are an expert at reformulating user messages based on conversation history.
    """

    _CONSTRUCT_QUERY_PROMPT_TEMPLATE = """
    Given the following chat history and the latest user message, reformulate the user's message
    to better reflect the context of the conversation.

    ### Guidelines
    1. The reformulated message should capture the context of the user's latest
    message, taking into account the previous messages in the chat history.
    2. Your response should contain ONLY the reformulated message.
    3. The reformulated message MUST have the same meaning and grammatical structure as the
    original user message. E.g. if the user message is a question, the reformulated message
    should also be a question, as if the user had asked it.

    ### Chat History:
    {chat_history}

    ### Latest User Message:
    {user_message}
    """

    def __init__(self,
                 cfg: DocRetrievalCfg,
                 vector_store_proxy: vs_core.VectorStoreProxy) -> None:

        _logger().info('Initializing DocRetrievalService with config: %s', cfg)

        self._cfg = cfg
        self._vector_store_proxy = vector_store_proxy
        self._helper_llm = ChatOllama(
            model=self._cfg.helper_llm.model_name,
            base_url=self._cfg.helper_llm.url,
            temperature=self._cfg.helper_llm.temperature
        )

    async def retrieve_context_docs(self,
                                    user_message: str,
                                    chat_history: list[dict[str, str]]
                                    ) -> list[dict[str, Any]]:
        """Retrieves context documents based on user message and chat history.

        Args:
            user_message: The latest message from the user.
            chat_history: List of previous messages in the conversation.
        """

        _logger().debug('Retrieving context documents for user message: %s and chat history: %s',
                        user_message,
                        chat_history)

        query = await self._construct_query(user_message, chat_history)

        _logger().debug('Constructed query for document retrieval: %s', query)

        retrieved_docs = await self._vector_store_proxy.retrieve_documents(
            query,
            self._cfg.max_context_docs,
            self._cfg.similarity_threshold)

        return [
            {'content': doc.page_content, 'metadata': doc.metadata}
            for doc in retrieved_docs
        ]

    async def _construct_query(self,
                               user_message: str,
                               chat_history: list[dict[str, str]]
                               ) -> str:
        """Constructs a query string using the helper LLM based on user message and chat history.

        Args:
            user_message: The latest message from the user.
            chat_history: List of previous messages in the conversation.
        """

        formatted_chat_history = self._format_chat_history(chat_history)

        messages = [
            SystemMessage(self._CONSTRUCT_QUERY_SYSTEM_PROMPT),
            HumanMessage(
                self._CONSTRUCT_QUERY_PROMPT_TEMPLATE.format(
                    chat_history=formatted_chat_history,
                    user_message=user_message
                )
            )
        ]

        response = await self._helper_llm.agenerate([messages])

        return response.generations[0][0].text.strip()

    def _format_chat_history(self,
                             chat_history: list[dict[str, str]]
                             ) -> str:
        """Formats the chat history into a string for prompt input.

        Args:
            chat_history: List of previous messages in the conversation.
        """

        history_lines = [
            f"{entry['role'].capitalize()}: {entry['content']}"
            for entry in chat_history
        ]

        return "\n".join(history_lines)
