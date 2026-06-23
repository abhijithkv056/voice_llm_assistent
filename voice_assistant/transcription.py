"""Speech-to-text using faster-whisper."""

from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from .config import Settings
from .logging_config import get_logger

logger = get_logger(__name__)


class Transcriber:
    """Wraps a Whisper model for audio-file transcription."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        logger.info("Loading Whisper model '%s'", settings.whisper_model_name)
        self._model = WhisperModel(
            settings.whisper_model_name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    def transcribe(self, file_path: Path | str) -> str:
        """Transcribe an audio file into stripped text."""
        segments, _ = self._model.transcribe(
            str(file_path), beam_size=self._settings.whisper_beam_size
        )
        return " ".join(segment.text for segment in segments).strip()
