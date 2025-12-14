"""Contains utilities used by the backend services."""
import dataclasses
from typing import Any
from typing import Dict
from typing import List
from typing import TypeAlias


@dataclasses.dataclass
class ChatMessage:
    """Represents a single chat message."""

    role: str
    content: str


@dataclasses.dataclass
class ChatHistory:
    """Contains history of messages in a chat session."""

    messages: List[ChatMessage]


@dataclasses.dataclass
class ContextDocument:
    """Represents a single document retrieved from the doc store."""

    title: str
    content: str


UnstructuredChatHistory: TypeAlias = List[Dict[str, Any]]

UnstructuredContextDocs: TypeAlias = List[Dict[str, Any]]


def chat_history_to_payload(chat_history: ChatHistory) -> UnstructuredChatHistory:
    """Converts chat history into json representation containing only core fields."""
    return [
        {
            'role': message.role,
            'content': message.content
        }
        for message in chat_history.messages
    ]


def context_docs_to_payload(context_docs: List[ContextDocument]) -> UnstructuredContextDocs:
    """Converts context docs into json representation."""

    return [
        {
            'title': doc.title,
            'content': doc.content
        }
        for doc in context_docs
    ]
