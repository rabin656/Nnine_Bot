import os
import pickle
import numpy as np
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

class SimpleVectorStore:
    """A simple in-memory vector store using cosine similarity."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.data = []  # List of dictionaries: {'vector': np.array, 'document': Document}

    def add_documents(self, documents):
        """Embeds and stores documents."""
        if not documents:
            return
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        
        for doc, vector in zip(documents, embeddings):
            self.data.append({
                'vector': np.array(vector),
                'document': doc
            })

    def save(self, filepath):
        """Saves the vector store to a pickle file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f)
            print(f"Vector store saved to {filepath}")

    def load(self, filepath):
        """Loads the vector store from a pickle file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Vector store loaded from {filepath}")
        else:
            print(f"Warning: {filepath} not found.")

    def similarity_search(self, query, k=3):
        """Finds the top k most similar documents."""
        if not self.data:
            return []
            
        query_vector = np.array(self.embedding_model.embed_query(query))
        query_norm = np.linalg.norm(query_vector)
        
        scores = []
        for item in self.data:
            doc_vector = item['vector']
            doc_norm = np.linalg.norm(doc_vector)
            
            if doc_norm == 0 or query_norm == 0:
                similarity = 0
            else:
                similarity = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
                
            scores.append((similarity, item['document']))
            
        # Sort by similarity in descending order
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k documents
        return [doc for score, doc in scores[:k]]

class LocalAIRagAgent:
    """Agent for RAG interactions using local models."""
    
    def __init__(self, llm_model="llama3.2", embedding_model_name="mxbai-embed-large", index_file="data/vector_store.pkl"):
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model_name
        self.index_file = index_file
        
        self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        self.llm = ChatOllama(model=self.llm_model)
        self.vector_store = SimpleVectorStore(self.embeddings)
        
        # Load index if it exists
        if os.path.exists(self.index_file):
            self.vector_store.load(self.index_file)

    def ingest_data(self, pdf_path, excel_path):
        """Loads data from PDF and Excel, chunks it, and rebuilds the index."""
        docs = []
        
        # Load PDF
        if os.path.exists(pdf_path):
            print(f"Loading {pdf_path}...")
            try:
                loader = PyPDFLoader(pdf_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading PDF: {e}")
        
        # Load Excel
        if os.path.exists(excel_path):
            print(f"Loading {excel_path}...")
            try:
                df = pd.read_excel(excel_path)
                for i, row in df.iterrows():
                    content = " | ".join([f"{k}: {v}" for k, v in row.items() if pd.notna(v)])
                    if content.strip():
                        docs.append(Document(page_content=content, metadata={"source": excel_path, "row": i}))
            except Exception as e:
                print(f"Error loading Excel: {e}")
                
        if not docs:
            print("No documents found.")
            return
            
        # Split text
        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # Re-initialize store (clear old data)
        self.vector_store = SimpleVectorStore(self.embeddings)
        print(f"Embedding {len(chunks)} chunks...")
        self.vector_store.add_documents(chunks)
        self.vector_store.save(self.index_file)
        print("Ingestion complete.")

    def ask(self, check_query):
        """Asks the agent a question."""
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(check_query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        template = """Imagine yourself as a helpful assistant of Nnine Solutions who's job is to answer the user question based on the context provided. If you don't have an answer based on the context, just say "Sorry, I don't know about this", Your answer must be in human language, the user must feel like he is talking to a real person. Answer the question in a way that is easy to understand and provides value to the user. Answer according to the context provided:
{context}

Question: {question}
"""
        prompt = PromptTemplate.from_template(template)
        formatted_prompt = prompt.format(context=context, question=check_query)
        
        try:
            response = self.llm.invoke(formatted_prompt)
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"
