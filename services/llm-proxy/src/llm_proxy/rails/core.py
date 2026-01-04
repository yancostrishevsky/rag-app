"""Contains core guardrails interfaces."""

from abc import ABC, abstractmethod
import dataclasses
from typing import Any


@dataclasses.dataclass
class LLMCallContext:
    """Context information about an LLM call."""

    user_message: str
    chat_history: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]


@dataclasses.dataclass
class GuardrailDecision:
    """Represents the decision made by a guardrail."""

    should_pass: bool
    reason: str | None = None


class Guardrail(ABC):
    """Interface for all input/output/retrieval guardrails."""

    @abstractmethod
    async def should_pass(self, llm_call_context: LLMCallContext) -> GuardrailDecision:
        """Tells whether the given LLM call context should pass the guardrail."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the guardrail."""
