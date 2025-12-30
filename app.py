import streamlit as st
from agent import LocalAIRagAgent

# Configuration
PAGE_TITLE = "Nnine Solutions Chatbot"
INDEX_FILE = "data/vector_store.pkl"

st.set_page_config(page_title=PAGE_TITLE, page_icon="🤖")
st.title("🤖 Nnine Solutions AI Assistant")

@st.cache_resource
def get_agent():
    return LocalAIRagAgent(index_file=INDEX_FILE)

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Nnine Solutions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.ask(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
