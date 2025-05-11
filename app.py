import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
import streamlit as st
import tempfile

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import voice_service as vs

# Constants
DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 10
STORAGE_PATH = 'rag/restaurant_file.txt'
EMBEDDING_MODEL = OpenAIEmbeddings()
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = ChatOpenAI()

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

class ConversationHistory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
    
    def add_exchange(self, user_input, assistant_response):
        self.history.append({
            "user": user_input,
            "assistant": assistant_response
        })
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
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file_path = temp_file.name
        with wave.open(temp_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))

    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True, None
        else:
            return False, temp_file_path
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False, None

def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def initialize_session_state():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationHistory(max_history=8)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("Voice Assistant Chat")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize Whisper model
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Load and index documents
    raw_docs = load_pdf_documents(STORAGE_PATH)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Record button
    if st.button("🎤 Start Recording"):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        
        with st.spinner("Listening..."):
            is_silent, temp_file_path = record_audio_chunk(audio, stream)
            
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        if not is_silent and temp_file_path:
            # Transcribe audio
            transcription = transcribe_audio(model, temp_file_path)
            os.remove(temp_file_path)
            
            # Display user message
            st.session_state.messages.append({"role": "user", "content": transcription})
            with st.chat_message("user"):
                st.markdown(transcription)
            
            # Check for end conversation
            end_phrases = ["end conversation", "stop", "goodbye", "bye", "exit", "quit"]
            if not any(phrase in transcription.lower() for phrase in end_phrases):
                # Generate response
                relevant_docs = find_related_documents(transcription)
                output = generate_answer(
                    transcription,
                    relevant_docs,
                    st.session_state.conversation.get_formatted_history()
                )
                
                if output:
                    output = output.lstrip()
                    st.session_state.conversation.add_exchange(transcription, output)
                    
                    # Display assistant message
                    st.session_state.messages.append({"role": "assistant", "content": output})
                    with st.chat_message("assistant"):
                        st.markdown(output)
                    
                    # Play audio response
                    vs.play_text_to_speech(output)
            
            st.rerun()

if __name__ == "__main__":
    main() 