"""Command-line voice assistant entrypoint."""

from __future__ import annotations

import contextlib

from .assistant import Assistant
from .audio import AudioRecorder
from .config import get_settings
from .logging_config import configure_logging, get_logger
from .tts import TextToSpeech

logger = get_logger(__name__)


def run() -> None:
    """Run the interactive voice loop until the user ends the conversation."""
    settings = get_settings()
    configure_logging(settings.log_level)

    print("\nVoice Assistant Started")
    print("Press Ctrl+C to exit")
    print("-" * 50)

    assistant = Assistant(settings)
    tts = TextToSpeech(settings)

    try:
        with AudioRecorder(settings) as recorder:
            while True:
                print("\nListening...")
                chunk_path = recorder.record_chunk()
                if chunk_path is None:
                    continue

                try:
                    transcription = assistant.transcriber.transcribe(chunk_path)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        chunk_path.unlink()

                if not transcription:
                    continue

                print(f"\nCustomer: {transcription}")

                if assistant.is_end_phrase(transcription):
                    print("\nEnding conversation. Goodbye!")
                    break

                reply = assistant.respond(transcription)
                if reply:
                    tts.speak(reply)
                    print(f"Assistant: {reply}\n")
                    print("-" * 50)
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    run()
