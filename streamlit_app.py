import streamlit as st
import os
import time
import pyaudio
import wave
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
import tempfile

# Constants
DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 10
STORAGE_PATH = 'rag/restaurant_file.txt'
TEMP_DIR = tempfile.gettempdir()

# Initialize Streamlit state
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False
if 'welcome_played' not in st.session_state:
    st.session_state.welcome_played = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = True
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'last_response_played' not in st.session_state:
    st.session_state.last_response_played = None

# Sidebar with Menu
with st.sidebar:
    st.title("🍽️ Our Menu")
    st.markdown("---")
    
    # Read and display menu from file
    try:
        with open(STORAGE_PATH, 'r') as file:
            menu_content = file.read()
            
        # Parse menu content
        menu_lines = menu_content.split('\n')
        restaurant_name = next((line.replace('Restaurant Name:', '').strip() 
                              for line in menu_lines if 'Restaurant Name:' in line), '')
        
        # Display restaurant name
        st.header(restaurant_name)
        
        # Find menu items and prices
        menu_items = []
        for line in menu_lines:
            if '-' in line and 'Rupees' in line:
                item, price = line.split('-')
                menu_items.append({
                    'item': item.strip(),
                    'price': price.replace('Rupees.', '₹').strip()
                })
        
        # Display menu items in a nice format
        st.subheader("📋 Menu Items")
        for item in menu_items:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{item['item']}**")
            with col2:
                st.write(f"*{item['price']}*")
                
        st.markdown("---")
        st.caption("Prices are inclusive of all taxes")
        
    except Exception as e:
        st.error(f"Error loading menu: {str(e)}")

# Initialize models and services
@st.cache_resource
def initialize_models():
    model_size = DEFAULT_MODEL_SIZE + ".en"
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    embedding_model = OllamaEmbeddings(model="mistral")
    document_vector_db = InMemoryVectorStore(embedding_model)
    language_model = OllamaLLM(model="mistral")
    return whisper_model, document_vector_db, language_model

# Load and process documents
@st.cache_resource
def load_and_process_documents():
    document_loader = TextLoader(STORAGE_PATH)
    documents = document_loader.load()
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_processor.split_documents(documents)
    return chunks

# Initialize everything
whisper_model, document_vector_db, language_model = initialize_models()
document_chunks = load_and_process_documents()
document_vector_db.add_documents(document_chunks)

# Streamlit UI
st.title("Restaurant Voice Assistant")

# Start conversation button
if not st.session_state.conversation_started:
    if st.button("Start Conversation"):
        st.session_state.conversation_started = True
        welcome_msg = "Hello! Welcome to our restaurant. I'm your virtual assistant. May I know your name?"
        st.session_state.conversation_history.append({"role": "assistant", "content": welcome_msg})
        vs.play_text_to_speech(welcome_msg)
        st.session_state.welcome_played = True
        st.rerun()

# Chat container
if st.session_state.conversation_started:
    chat_container = st.container()
    with chat_container:
        # Display chat messages
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

    # Audio recording function
    def record_audio():
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 10
        WAVE_OUTPUT_FILENAME = os.path.join(TEMP_DIR, f"temp_audio_{int(time.time())}.wav")
        
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return WAVE_OUTPUT_FILENAME

    # Generate response
    def generate_response(query, context_documents):
        PROMPT_TEMPLATE = """
        You are an voice assistant. You interact with customer calls and provide information from the context provided.
        In your conversation ensure following rules:
        1. If customer has provided their name, use it in your response.
        2. If customer is asking for a menu, provide the menu information from the context provided. Only tell the customer the menu items and not the prices unless they ask for it.
        3. If customer is asking for the location, provide the location information from the context provided.
        4. Keep responses concise and natural.

        Previous Conversation: {chat_history}
        Current Query: {user_query}
        Restaurant Information: {document_context}
        Answer:
        """
        
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        chat_history = "\n".join([f"{'Customer' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                                 for msg in st.session_state.conversation_history[-5:]])
        
        conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        response_chain = conversation_prompt | language_model
        return response_chain.invoke({
            "user_query": query,
            "document_context": context_text,
            "chat_history": chat_history
        })

    # Audio recording button and processing
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.processing_complete:
            button_label = "⏺️ Start Convo" if not st.session_state.recording else "🎤 Start recording"
            if st.button(button_label):
                if not st.session_state.recording:
                    st.session_state.recording = True
                    st.rerun()
                else:
                    # First update UI to show processing state
                    st.session_state.recording = False
                    st.session_state.processing_complete = False
                    
                    # Record audio
                    audio_file = record_audio()
                    
                    try:
                        # Step 1: Transcribe audio
                        segments, _ = whisper_model.transcribe(audio_file)
                        transcription = ' '.join(segment.text for segment in segments)
                        
                        # Step 2: Generate response
                        relevant_docs = document_vector_db.similarity_search(transcription)
                        response = generate_response(transcription, relevant_docs)
                        
                        # Step 3: Update conversation history
                        st.session_state.conversation_history.append({"role": "user", "content": transcription})
                        if response:
                            response = response.lstrip()
                            st.session_state.conversation_history.append({"role": "assistant", "content": response})
                            st.session_state.current_response = response
                            st.session_state.last_response_played = None  # Reset the last played response
                        
                        # Clean up
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                        
                        # Step 4: Enable the button for next interaction
                        st.session_state.recording = False
                        st.session_state.processing_complete = True
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                        st.session_state.recording = False
                        st.session_state.processing_complete = True
                    
                    st.rerun()
        else:
            st.button("Processing...", disabled=True)

    # Play audio response after UI is updated
    if (st.session_state.current_response and 
        st.session_state.last_response_played != st.session_state.current_response):
        try:
            vs.play_text_to_speech(st.session_state.current_response)
            st.session_state.last_response_played = st.session_state.current_response
            st.session_state.current_response = None
        except Exception as e:
            st.error(f"Error playing audio: {str(e)}")

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True) 