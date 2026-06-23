"""Microphone capture and silence detection."""

from __future__ import annotations

import contextlib
import tempfile
import wave
from pathlib import Path

import numpy as np
import pyaudio
from scipy.io import wavfile

from .config import Settings
from .logging_config import get_logger

logger = get_logger(__name__)


def is_silence(data: np.ndarray, threshold: int) -> bool:
    """Return ``True`` if the audio samples never exceed ``threshold``."""
    if data.size == 0:
        return True
    return bool(np.max(np.abs(data)) <= threshold)


class AudioRecorder:
    """Records fixed-length audio chunks from the default input device.

    Use as a context manager so the PyAudio resources are always released::

        with AudioRecorder(settings) as recorder:
            path = recorder.record_chunk()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._audio: pyaudio.PyAudio | None = None

    def __enter__(self) -> "AudioRecorder":
        self._audio = pyaudio.PyAudio()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._audio is not None:
            self._audio.terminate()
            self._audio = None

    def _require_audio(self) -> pyaudio.PyAudio:
        if self._audio is None:
            raise RuntimeError("AudioRecorder must be used as a context manager")
        return self._audio

    def record_chunk(self) -> Path | None:
        """Record one chunk; return its WAV path, or ``None`` if it was silent.

        The returned file is the caller's responsibility to delete.
        """
        audio = self._require_audio()
        settings = self._settings

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=settings.sample_rate,
            input=True,
            frames_per_buffer=settings.frames_per_buffer,
        )
        try:
            num_reads = int(
                settings.sample_rate
                / settings.frames_per_buffer
                * settings.chunk_length_seconds
            )
            frames = [
                stream.read(settings.frames_per_buffer, exception_on_overflow=False)
                for _ in range(num_reads)
            ]
        finally:
            stream.stop_stream()
            stream.close()

        temp_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        with wave.open(str(temp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(settings.sample_rate)
            wf.writeframes(b"".join(frames))

        try:
            _, data = wavfile.read(str(temp_path))
        except Exception:
            logger.exception("Failed to read recorded audio chunk")
            _safe_unlink(temp_path)
            return None

        if is_silence(data, settings.silence_threshold):
            _safe_unlink(temp_path)
            return None
        return temp_path


def _safe_unlink(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
