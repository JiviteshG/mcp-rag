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
    # Note: We did not use an embedding model here, chromadb will use the default embedding model to generate embeddings for the documents
    for doc in documents:
        collection.add(
            documents=[doc.text],
            metadatas=[doc.metadata],
            ids=[doc.doc_id]
        )

    final_count = collection.count()
    print(f"Final document count in collection '{collection_name}': {final_count}")
    return documents

def query_documents(query: str, n_results: int = 2, collection_name: str = COLLECTION_NAME) -> str:
    chroma_client = get_chromadb_client()
    collection = chroma_client.get_collection(name=collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    if len(results["documents"]) == 0 or not results["documents"][0]:
        return "No documents found."

    # Format the results for better readability
    formatted_results = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
    distances = results["distances"][0] if results["distances"] else [0] * len(documents)

    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        result_text = f"-- Result {i + 1} --\n"
        result_text += f"Document content: {doc}\n"
        result_text += f"Metadata: {metadata}\n"
        result_text += f"Distance: {distance}\n"
        formatted_results.append(result_text)

    response = f"FOund {len(formatted_results)} results for query: '{query}'\n\n" + "\n".join(formatted_results) 
    response += "\n".join(formatted_results) 

    return response
def main():
    init_chromadb()

    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    print("Initialized ChromaDB and LlamaParse API Key!..")

    documents = ingest_data_directory(LLAMA_CLOUD_API_KEY, COLLECTION_NAME, DATA_DIR)
    print(f"Ingested {len(documents)} documents.")

    response = query_documents("What is TradingAgents?", n_results=2, collection_name=COLLECTION_NAME)
    print(response)


if __name__ == "__main__":
    main()
