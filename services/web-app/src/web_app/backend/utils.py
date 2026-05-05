"""Contains utilities used by the backend services."""
import dataclasses
from typing import Any
from typing import TypeAlias

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
    retrieval_score: float | None = None


@dataclasses.dataclass
class ContextRetrievalResult:
    """Contains retrieval results along with retrieval debug information."""

    context_docs: list[ContextDocument]
    rewritten_query: str


@dataclasses.dataclass
class InputCheckResult:
    """Result of input safety, or relevance check."""

    is_ok: bool
    reason: str | None
    raw_response: str | None = None


@dataclasses.dataclass
class TurnDebugInfo:
    """Debug trace for a single chat turn."""

    user_message: str
    safety_check: InputCheckResult | None = None
    relevance_check: InputCheckResult | None = None
    rewritten_query: str | None = None
    context_docs: list[ContextDocument] = dataclasses.field(default_factory=list)
    prompt_preview: str | None = None
    final_response: str | None = None


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
            'retrieval_score': doc.retrieval_score,
        }
        for doc in context_docs
    ]
