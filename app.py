import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration
PAGE_TITLE = "Nnine Solutions Chatbot"
INDEX_FOLDER = "faiss_index_local"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"

st.set_page_config(page_title=PAGE_TITLE, page_icon="🤖")
st.title("🤖 Nnine Solutions AI Assistant")

@st.cache_resource
def load_chain():
    """Loads the RAG chain."""
    if not os.path.exists(INDEX_FOLDER):
        return None
    
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
        
        llm = ChatOllama(model=LLM_MODEL)
        
        template = """Answer the question based ONLY on the following context:
        {context}
        
        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        return chain
    except Exception as e:
        st.error(f"Error loading chain: {e}")
        return None

chain = load_chain()

if not chain:
    st.warning("⚠️ Knowledge base not found. Please run `python vector_store.py` first.")
else:
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
                try:
                    response = chain.run(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
