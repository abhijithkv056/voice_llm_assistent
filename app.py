import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


import voice_service as vs
#details

DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 10

PROMPT_TEMPLATE = """
You are an voice assistant. You interact with customer calls and provide information from the context provided.
In your convversation ensure following rules:
1. Only in the first conversation, you should introduce yourself and ask for the customer's name if customer has not provided it.Then ask if he would like to order anything if he hasnt ordered anything yet.
2. If customer has provided their name, use it in your response.
3. If customer is asking for a menu, provide the menu information from the context provided. Only tell the customer the menu items and not the prices unless they ask for it.
4. If customer is asking for the location, provide the location information from the context provided.


Previous Conversation:{chat_history}
Current Query: {user_query} 
Restaurant Information: {document_context} 
Answer:
"""
STORAGE_PATH = 'rag/restaurant_file.txt'
EMBEDDING_MODEL = OllamaEmbeddings(model="mistral")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="mistral")

class ConversationHistory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
    
    def add_exchange(self, user_input, assistant_response):
        self.history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        # Keep only the last max_history exchanges
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_formatted_history(self):
        formatted = ""
        for exchange in self.history:
            formatted += f"Customer: {exchange['user']}\n"
            formatted += f"Assistant: {exchange['assistant']}\n"
        return formatted

def load_pdf_documents(file_path):
    document_loader = TextLoader(file_path)
    documents = document_loader.load()
    return documents

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents, chat_history):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({
        "user_query": user_query, 
        "document_context": context_text,
        "chat_history": chat_history
    })

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False

def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def main():
    print("\nVoice Assistant Started")
    print("Press Ctrl+C to exit")
    print("-" * 50)
    
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    raw_docs = load_pdf_documents(STORAGE_PATH)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)    

    audio = pyaudio.PyAudio()
    conversation = ConversationHistory(max_history=8)  # Keep last 8 exchanges

    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"
            
            # Open stream for recording
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
            
            # Record audio chunk
            print("\nListening...")
            if not record_audio_chunk(audio, stream):
                # Close stream after recording
                stream.stop_stream()
                stream.close()
                
                # Transcribe audio
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)
                print("\nCustomer: {}".format(transcription))
                
                # Check if user wants to end conversation
                end_phrases = ["end conversation", "stop", "goodbye", "bye", "exit", "quit"]
                if any(phrase in transcription.lower() for phrase in end_phrases):
                    print("\nEnding conversation. Goodbye!")
                    break
                
                # Get relevant documents and generate response
                relevant_docs = find_related_documents(transcription)
                output = generate_answer(
                    transcription, 
                    relevant_docs,
                    conversation.get_formatted_history()
                )
                
                if output:
                    output = output.lstrip()
                    # Add the exchange to conversation history
                    conversation.add_exchange(transcription, output)
                    # Play and print response
                    vs.play_text_to_speech(output)
                    print("Assistant: {}\n".format(output))
                    print("-" * 50)
            else:
                # Close stream if it was silence
                stream.stop_stream()
                stream.close()

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        audio.terminate()

if __name__ == "__main__":
    main()
