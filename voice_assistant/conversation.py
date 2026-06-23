"""Conversation history tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Exchange:
    """A single user/assistant turn."""

    user: str
    assistant: str


class ConversationHistory:
    """Rolling window of the most recent conversation exchanges."""

    def __init__(self, max_history: int = 8) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self._exchanges: deque[Exchange] = deque(maxlen=max_history)

    def add_exchange(self, user_input: str, assistant_response: str) -> None:
        """Record a user query and the assistant's reply."""
        self._exchanges.append(Exchange(user=user_input, assistant=assistant_response))

    def formatted(self) -> str:
        """Return the history rendered as prompt-ready text."""
        return "\n".join(
            f"Customer: {exchange.user}\nAssistant: {exchange.assistant}"
            for exchange in self._exchanges
        )

    def __len__(self) -> int:
        return len(self._exchanges)
