import streamlit as st
import audio_manager
from agent import LocalAIRagAgent

# --- Configuration ---
PAGE_TITLE = "Nnine Solutions Chatbot"
PAGE_ICON = "🤖"
INDEX_FILE = "data/vector_store.pkl"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.title(f"{PAGE_ICON} N9 Solutions Assistant")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---

@st.cache_resource
def get_agent():
    """Initializes the RAG agent (cached)."""
    return LocalAIRagAgent(index_file=INDEX_FILE)

def display_chat_history():
    """Renders all previous messages."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def handle_conversation(user_text, is_voice_input=False):
    """
    Main Logic:
    1. Display user message.
    2. Get Agent response.
    3. Display Agent response.
    4. IF input was voice -> Generate & Play Audio.
    """
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2. Agent Response
    agent = get_agent()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = agent.ask(user_text)
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        # 3. Audio Response (Voice Mode Only)
        if is_voice_input:
            with st.spinner("Generating Voice..."):
                audio_path = audio_manager.generate_audio_from_text(response_text)
                if audio_path:
                    st.audio(audio_path, format="audio/mp3", autoplay=True)

# --- Main App Flow ---

display_chat_history()

# Layout for inputs
st.divider()
audio_value = st.audio_input("🎤 Record your question")
text_value = st.chat_input("Ask about Nnine Solutions...")

# Process Input
if audio_value:
    with st.spinner("Transcribing..."):
        transcribed_text = audio_manager.transcribe_audio(audio_value)
    
    if transcribed_text:
        handle_conversation(transcribed_text, is_voice_input=True)
    else:
        st.error("Could not transcribe audio. Please try again.")

elif text_value:
    handle_conversation(text_value, is_voice_input=False)
