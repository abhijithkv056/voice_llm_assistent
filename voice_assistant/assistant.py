"""High-level assistant orchestration shared by all entrypoints."""

from __future__ import annotations

from .config import Settings, get_settings
from .conversation import ConversationHistory
from .rag import RagService
from .transcription import Transcriber

END_PHRASES = ("end conversation", "stop", "goodbye", "bye", "exit", "quit")


class Assistant:
    """Combines transcription, retrieval, and conversation memory.

    This is the single source of truth for the conversational logic; the CLI
    and web frontends only handle I/O.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.transcriber = Transcriber(self._settings)
        self.rag = RagService(self._settings)
        self.history = ConversationHistory(max_history=self._settings.max_history)

    @staticmethod
    def is_end_phrase(text: str) -> bool:
        """Return ``True`` if the text signals the customer wants to hang up."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in END_PHRASES)

    def respond(self, user_query: str) -> str:
        """Generate a reply for ``user_query`` and record the exchange."""
        reply = self.rag.answer(user_query, self.history.formatted())
        self.history.add_exchange(user_query, reply)
        return reply
