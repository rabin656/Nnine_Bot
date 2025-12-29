import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configuration
PDF_FILE = "nnine_chatbot_train_data.pdf"
EXCEL_FILE = "nnine_chatbot_training_data.xlsx"
INDEX_FOLDER = "faiss_index_local"
EMBEDDING_MODEL = "mxbai-embed-large"

def load_documents():
    """Loads documents from PDF and Excel files."""
    docs = []
    
    # Load PDF
    if os.path.exists(PDF_FILE):
        print(f"Loading {PDF_FILE}...")
        try:
            loader = PyPDFLoader(PDF_FILE)
            docs.extend(loader.load())
            print(f"Loaded {len(docs)} pages from PDF.")
        except Exception as e:
            print(f"Error loading PDF: {e}")
    else:
        print(f"Warning: {PDF_FILE} not found.")

    # Load Excel
    if os.path.exists(EXCEL_FILE):
        print(f"Loading {EXCEL_FILE}...")
        try:
            df = pd.read_excel(EXCEL_FILE)
            excel_docs = []
            for i, row in df.iterrows():
                # Convert row to string format
                content = " | ".join([f"{k}: {v}" for k, v in row.items() if pd.notna(v)])
                if content.strip():
                    excel_docs.append(Document(page_content=content, metadata={"source": EXCEL_FILE, "row": i}))
            docs.extend(excel_docs)
            print(f"Loaded {len(excel_docs)} rows from Excel.")
        except Exception as e:
            print(f"Error loading Excel: {e}")
    else:
        print(f"Warning: {EXCEL_FILE} not found.")
        
    return docs

def create_index():
    """Creates and saves the FAISS index."""
    documents = load_documents()
    if not documents:
        print("No documents to process.")
        return

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Generating embeddings using {EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(INDEX_FOLDER)
        print(f"Success! Vector store saved to '{INDEX_FOLDER}'.")
    except Exception as e:
        print(f"Error creating vector store: {e}")

if __name__ == "__main__":
    create_index()
