# Restaurant Voice Assistant — Streamlit web frontend.
#
# NOTE: This image runs the web UI, which captures audio from the *server's*
# input device via PyAudio. For real microphone input the container must be
# granted access to the host's audio hardware (e.g. `--device /dev/snd`).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System libraries required by PyAudio (portaudio) and faster-whisper (ffmpeg).
RUN apt-get update && apt-get install -y --no-install-recommends \
        portaudio19-dev \
        libportaudio2 \
        ffmpeg \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY voice_assistant ./voice_assistant
COPY rag ./rag

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "voice_assistant/web.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
