"""Application configuration.

Settings are loaded from environment variables (and an optional ``.env``
file) so the app can be configured per-environment without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


class Settings(BaseSettings):
    """Runtime configuration, overridable via environment variables.

    Every field maps to an upper-cased environment variable, e.g.
    ``OLLAMA_MODEL`` overrides :attr:`ollama_model`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Language model / RAG (Ollama) ---
    ollama_model: str = Field(
        default="mistral",
        description="Ollama model used for both embeddings and generation.",
    )
    ollama_base_url: str | None = Field(
        default=None,
        description="Base URL of the Ollama server. Defaults to Ollama's own default.",
    )
    knowledge_base_path: Path = Field(
        default=PROJECT_ROOT / "rag" / "restaurant_file.txt",
        description="Path to the text document used as the RAG knowledge base.",
    )
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # --- Speech-to-text (Whisper) ---
    whisper_model_size: str = Field(default="medium")
    whisper_language: str = Field(default="en")
    whisper_device: str = Field(default="cpu")
    whisper_compute_type: str = Field(default="int8")
    whisper_beam_size: int = Field(default=7, gt=0)

    # --- Audio capture ---
    sample_rate: int = Field(default=16000, gt=0)
    frames_per_buffer: int = Field(default=1024, gt=0)
    chunk_length_seconds: int = Field(default=10, gt=0)
    silence_threshold: int = Field(default=3000, ge=0)

    # --- Text-to-speech ---
    tts_language: str = Field(default="en")
    tts_slow: bool = Field(default=False)

    # --- Conversation ---
    max_history: int = Field(default=8, gt=0)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    @property
    def whisper_model_name(self) -> str:
        """Whisper model identifier, with the language suffix for English."""
        if self.whisper_language == "en":
            return f"{self.whisper_model_size}.en"
        return self.whisper_model_size


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
