# Restaurant Voice Assistant V1

A voice-enabled restaurant assistant that helps customers with menu information and order taking using natural language processing.

## Features

- Voice-based interaction with customers
- Real-time speech-to-text conversion
- Natural language understanding
- Menu information retrieval
- Order taking capabilities
- Interactive web interface

## Prerequisites

- Python 3.10 or higher
- Ollama installed with Mistral model
- Working microphone
- Speakers or headphones

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd voice_assistant_llm
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

5. Make sure Ollama is running and the Mistral model is pulled:
```bash
ollama pull mistral
```

## Running the Application

You can run the application in two ways:

### 1. Command Line Interface
```bash
python app.py
```

### 2. Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```

The web interface provides:
- Interactive chat interface
- Menu display in sidebar
- Easy-to-use voice recording buttons
- Visual conversation history

## Usage

1. Start the application using either method above
2. Click "Start Conversation" to begin
3. Use the microphone button to speak your requests
4. View the menu in the sidebar
5. Interact naturally with the assistant

## Project Structure

- `app.py`: Command-line interface version
- `streamlit_app.py`: Web interface version
- `voice_service.py`: Voice processing utilities
- `rag/`: Contains restaurant information files
- `requirements.txt`: Package dependencies

## Troubleshooting

- Ensure your microphone is properly connected and has necessary permissions
- Check if Ollama is running (`ollama serve`)
- Verify that the Mistral model is downloaded
- Make sure all dependencies are installed correctly

## License

[Your License]

## Contributing

[Contribution guidelines]


Internal note:

So how it work:

1. Pyaudio --> access your mic to get a mono audio, 16Khz file -> then save it as .wave which Whisper can read and process
2. whisper model --> takes the wave file and returns the transcription
3. LLM -> takes the transcription + context (via RAG) and generates a response based on prompt template
4. response is saved to an mp3 file via gTTS
5. the mp3 file is payed back ny pygame library
