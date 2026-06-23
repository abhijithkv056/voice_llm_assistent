"""Streamlit web frontend for the voice assistant.

Run with::

    streamlit run voice_assistant/web.py
"""

from __future__ import annotations

import contextlib

import streamlit as st

from .assistant import Assistant
from .audio import AudioRecorder
from .config import Settings, get_settings
from .conversation import ConversationHistory
from .logging_config import configure_logging
from .rag import RagService
from .transcription import Transcriber
from .tts import TextToSpeech


@st.cache_resource
def _load_transcriber(_settings: Settings) -> Transcriber:
    return Transcriber(_settings)


@st.cache_resource
def _load_rag(_settings: Settings) -> RagService:
    return RagService(_settings)


@st.cache_resource
def _load_tts(_settings: Settings) -> TextToSpeech:
    return TextToSpeech(_settings)


def _record_once(settings: Settings, transcriber: Transcriber) -> str | None:
    with AudioRecorder(settings) as recorder:
        with st.spinner("Listening..."):
            chunk_path = recorder.record_chunk()
    if chunk_path is None:
        return None
    try:
        return transcriber.transcribe(chunk_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            chunk_path.unlink()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    st.title("Restaurant Voice Assistant")

    transcriber = _load_transcriber(settings)
    rag = _load_rag(settings)
    tts = _load_tts(settings)

    if "history" not in st.session_state:
        st.session_state.history = ConversationHistory(max_history=settings.max_history)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button("🎤 Start Recording"):
        transcription = _record_once(settings, transcriber)
        if not transcription:
            st.warning("No speech detected. Please try again.")
            return

        st.session_state.messages.append({"role": "user", "content": transcription})

        if Assistant.is_end_phrase(transcription):
            st.session_state.messages.append(
                {"role": "assistant", "content": "Ending conversation. Goodbye!"}
            )
            st.rerun()

        history = st.session_state.history
        reply = rag.answer(transcription, history.formatted())
        history.add_exchange(transcription, reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        tts.speak(reply)
        st.rerun()


if __name__ == "__main__":
    main()
