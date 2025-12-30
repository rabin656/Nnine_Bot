from agent import LocalAIRagAgent

# Configuration
PDF_FILE = "data/nnine_chatbot_train_data.pdf"
EXCEL_FILE = "data/nnine_chatbot_training_data.xlsx"
INDEX_FILE = "data/vector_store.pkl"

if __name__ == "__main__":
    print("Initialize Agent...")
    agent = LocalAIRagAgent(index_file=INDEX_FILE)
    
    print("Starting ingestion...")
    agent.ingest_data(PDF_FILE, EXCEL_FILE)
    print("Done!")
