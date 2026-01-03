from agent import LocalAIRagAgent

# Configuration
EXCEL_FILE = "data/nnine-training-2026-01-01-2.xlsx"
INDEX_FILE = "data/vector_store.pkl"

if __name__ == "__main__":
    print("Initialize Agent...")
    agent = LocalAIRagAgent(index_file=INDEX_FILE)
    
    print("Starting ingestion...")
    agent.ingest_data(EXCEL_FILE)
    print("Done!")
