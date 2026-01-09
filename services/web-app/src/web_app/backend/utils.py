"""Contains utilities used by the backend services."""
import dataclasses
from typing import TypeAlias
from typing import Any

import pydantic


class EndpointConnectionCfg(pydantic.BaseModel):
    """Configuration for connecting to a backend endpoint."""

    url: str
    connection_timeout: float


@dataclasses.dataclass
class ChatMessage:
    """Represents a single chat message."""

    role: str
    content: str


@dataclasses.dataclass
class ChatHistory:
    """Contains history of messages in a chat session."""

    messages: list[ChatMessage]


@dataclasses.dataclass
class ContextDocument:
    """Represents a single document retrieved from the doc store."""

    content: str
    metadata: dict[str, Any]


@dataclasses.dataclass
class InputCheckResult:
    """Result of input safety, or relevance check."""

    is_ok: bool
    reason: str | None = None


UnstructuredChatHistory: TypeAlias = list[dict[str, str]]

UnstructuredContextDocs: TypeAlias = list[dict[str, Any]]


def chat_history_to_payload(chat_history: ChatHistory) -> UnstructuredChatHistory:
    """Converts chat history into json representation containing only core fields."""
    return [
        {
            'role': message.role,
            'content': message.content
        }
        for message in chat_history.messages
    ]


def context_docs_to_payload(context_docs: list[ContextDocument]) -> UnstructuredContextDocs:
    """Converts context docs into json representation."""

    return [
        {
            'content': doc.content,
            'metadata': doc.metadata,
        }
        for doc in context_docs
    ]
