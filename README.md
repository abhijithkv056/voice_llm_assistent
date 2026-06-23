# Restaurant Voice Assistant

A voice-enabled restaurant assistant. It listens to a customer over the
microphone, transcribes the speech, answers questions about the restaurant
using a retrieval-augmented generation (RAG) pipeline backed by a local
Ollama model, and replies with synthesized speech.

## How it works

1. **Capture** – PyAudio records a mono, 16 kHz audio chunk and writes a WAV file.
2. **Transcribe** – `faster-whisper` converts the WAV file to text.
3. **Answer** – the transcription plus relevant context (retrieved from the
   knowledge base via embeddings) is passed to an Ollama LLM using a prompt template.
4. **Speak** – the reply is synthesized to MP3 with gTTS and played back via pygame.

## Project structure

```
voice_assistant/
├── __init__.py
├── __main__.py        # `python -m voice_assistant` → CLI
├── config.py          # Settings loaded from env / .env (pydantic-settings)
├── logging_config.py  # Centralized logging setup
├── conversation.py    # Rolling conversation history
├── audio.py           # Microphone capture + silence detection
├── transcription.py   # Whisper speech-to-text
├── tts.py             # gTTS + pygame text-to-speech
├── prompts.py         # Prompt template
├── rag.py             # Document indexing + grounded answer generation
├── assistant.py       # Orchestration shared by all frontends
├── cli.py             # Command-line entrypoint
└── web.py             # Streamlit entrypoint
rag/restaurant_file.txt  # Knowledge base
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the `mistral` model pulled
- A working microphone and speakers

## Installation

```bash
git clone <repository-url>
cd voice_assistant_llm

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # or: pip install -e .

ollama pull mistral
```

> **macOS/Linux:** PyAudio needs PortAudio installed
> (`brew install portaudio` or `sudo apt-get install portaudio19-dev`).

## Configuration

All settings have sensible defaults and can be overridden via environment
variables or a `.env` file. Copy the example to get started:

```bash
cp .env.example .env
```

See [.env.example](.env.example) for the full list (Ollama model/URL, Whisper
model size, audio parameters, logging level, etc.).

## Running

Make sure Ollama is running (`ollama serve`).

### Command line

```bash
python -m voice_assistant        # or: voice-assistant  (after `pip install -e .`)
```

Speak when you see `Listening...`. Say "goodbye", "stop", or "exit" to end.

### Web interface

```bash
streamlit run voice_assistant/web.py
```

## Docker

The provided [Dockerfile](Dockerfile) builds the Streamlit frontend:

```bash
docker build -t voice-assistant .
docker run -p 8501:8501 --device /dev/snd voice-assistant
```

> Audio is captured on the **server** side, so the container needs access to
> the host's audio hardware (`--device /dev/snd` on Linux). Ollama must be
> reachable from inside the container — set `OLLAMA_BASE_URL` accordingly.

## License

MIT
