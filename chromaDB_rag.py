import chromadb
from chromadb.config import Settings
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage


class Pipeline:
    def __init__(self):
        self.client = None
        self.collection = None

    async def on_startup(self):
        # Set the OpenAI API key
        import os
        os.environ["OPENAI_API_KEY"] = "your-api-key-here"

        # Initialize Chroma client
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
        
        # Load documents and create an index/collection in ChromaDB
        documents = self.load_documents("./data")  # Load documents from your data directory
        self.collection = self.client.create_collection("knowledge_base")

        # Add documents to ChromaDB collection
        for doc in documents:
            self.collection.add(
                documents=[doc["text"]],
                metadatas=[doc["metadata"]],
                ids=[doc["id"]],
            )

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def load_documents(self, directory: str) -> List[dict]:
        # Placeholder function to load documents from a directory (you can customize it based on your data format)
        import os
        documents = []
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), "r") as file:
                text = file.read()
                documents.append({"id": filename, "text": text, "metadata": {"source": filename}})
        return documents

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # This function retrieves relevant documents from the ChromaDB collection based on the user's query

        print(messages)
        print(user_message)

        # Query the collection with the user's message (assuming it's converted to embeddings)
        query_results = self.collection.query(query_embeddings=user_message, n_results=5)

        # Process the results and return a response (customize this based on your needs)
        response = self.synthesize_response(query_results)
        
        return response

    def synthesize_response(self, query_results: dict) -> str:
        # Synthesize the response based on the query results (this is a placeholder)
        # You would typically format and combine relevant document excerpts here
        response = "Relevant information:\n"
        for result in query_results["documents"]:
            response += f"- {result}\n"
        return response
