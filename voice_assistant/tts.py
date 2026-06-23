"""Text-to-speech playback using gTTS and pygame."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import pygame
from gtts import gTTS

from .config import Settings
from .logging_config import get_logger

logger = get_logger(__name__)


class TextToSpeech:
    """Synthesizes text to speech and plays it through the default device."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def synthesize(self, text: str) -> Path:
        """Render ``text`` to a temporary MP3 file and return its path."""
        tts = gTTS(
            text=text,
            lang=self._settings.tts_language,
            slow=self._settings.tts_slow,
        )
        path = Path(tempfile.mkstemp(suffix=".mp3")[1])
        tts.save(str(path))
        return path

    def speak(self, text: str) -> None:
        """Synthesize ``text`` and block until playback finishes."""
        if not text.strip():
            return

        audio_path = self.synthesize(text)
        try:
            pygame.mixer.init()
            try:
                pygame.mixer.music.load(str(audio_path))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            finally:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except pygame.error:
            logger.exception("Audio playback failed")
        finally:
            with contextlib.suppress(FileNotFoundError):
                audio_path.unlink()
