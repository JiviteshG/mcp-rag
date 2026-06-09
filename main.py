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
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = client.get_or_create_collection(name="mcp_rag_collection")    

def get_chromadb_client():
    return chromadb.PersistentClient(path="./chroma_db")

def ingest_data_directory():
    chroma_client = get_chromadb_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), api_url=os.getenv("LLAMA_CLOUD_API_URL"), result_type="text")
    
    file_extractor = {
        ".pdf": parser
        }
    documents = SimpleDirectoryReader(DATA_DIR, file_extractor=file_extractor).load_data()

    return documents

def main():
    init_chromadb()

    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    print("Hello from chromadb!")

    documents = ingest_data_directory()
    print(f"Ingested {len(documents)} documents.")


if __name__ == "__main__":
    main()
