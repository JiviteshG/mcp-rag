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
    chroma_client.delete_collection(name=collection_name)  # Delete the collection if it exists
    collection = chroma_client.get_or_create_collection(name=collection_name)

    parser = LlamaParse(api_key=llama_cloud_api_key, result_type="text")

    # The file extracter accepts pdf and docx files, and uses the LlamaParse parser to extract text from them
    file_extractor = {
        ".pdf": parser,
        ".docx": parser,
        }
    documents = SimpleDirectoryReader(data_dir, file_extractor=file_extractor).load_data()

    # Added documents to chromadb vector database
    for doc in documents:
        collection.add(
            documents=[doc.text],
            metadatas=[doc.metadata],
            ids=[doc.doc_id]
        )

    final_count = collection.count()
    print(f"Final document count in collection '{collection_name}': {final_count}")
    return documents

def main():
    init_chromadb()

    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    print("Hello from chromadb!")

    documents = ingest_data_directory(LLAMA_CLOUD_API_KEY, COLLECTION_NAME, DATA_DIR)
    print(f"Ingested {len(documents)} documents.")


if __name__ == "__main__":
    main()
