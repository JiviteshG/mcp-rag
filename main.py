import os
import chromadb
from dotenv import load_dotenv
from chromadb import Client
from chromadb.config import Settings
from llama_index.core import SimpleDirectoryReader
from llama_cloud_services import LlamaParse

load_dotenv()

PERSISTENCE_DIR = "./chroma_db"
COLLECTION_NAME = "mcp_rag_collection"
DATA_DIR = "./papers"

def init_chromadb():
    client = chromadb.PersistentClient(path=PERSISTENCE_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)   
    return collection 

def get_chromadb_client():
    return chromadb.PersistentClient(path=PERSISTENCE_DIR)

def ingest_data_directory(llama_cloud_api_key, collection_name, data_dir):
    chroma_client = get_chromadb_client()
    collection = chroma_client.get_collection(name=collection_name)

    parser = LlamaParse(api_key=llama_cloud_api_key, result_type="text")
    
    file_extractor = {
        ".pdf": parser,
        ".docx": parser,
        }
    documents = SimpleDirectoryReader(data_dir, file_extractor=file_extractor).load_data()

    return documents

def main():
    init_chromadb()

    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    print("Hello from chromadb!")

    documents = ingest_data_directory(LLAMA_CLOUD_API_KEY, COLLECTION_NAME, DATA_DIR)
    print(f"Ingested {len(documents)} documents.")


if __name__ == "__main__":
    main()
