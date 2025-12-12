"""Contains utilities used by the backend services."""

from typing import TypeAlias
from typing import Dict
from typing import List

ChatHistoryType: TypeAlias = List[Dict[str, str]]


def sanitize_chat_history(chat_history: ChatHistoryType) -> ChatHistoryType:
    """Sanitizes the chat history to ensure it contains only relevant fields."""
    return [
        {
            'role': item['role'],
            'content': item['content']
        }
        for item in chat_history
    ]
