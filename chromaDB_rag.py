import chromadb
import openai
from sentence_transformers import SentenceTransformer
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage

class Pipeline:
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.openai_client = None

    async def on_startup(self):
        # Initialize components
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # ChromaDB configuration
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.chroma_client.get_collection("scripts")
        except ValueError:
            raise RuntimeError("Collection 'scripts' not found in ChromaDB")
            
        # OpenAI client setup
        self.openai_client = openai.OpenAI(api_key="your-api-key-here")

    async def on_shutdown(self):
        # Cleanup resources
        if self.openai_client:
            self.openai_client.close()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Embed the user query
            query_embedding = self.embedding_model.encode(user_message).tolist()
            
            # Retrieve context from ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=["metadatas"]
            )
            
            # Process results
            context = self._process_results(results)
            
            # Generate response with OpenAI
            return self._generate_response(user_message, context)
            
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def _process_results(self, results: dict) -> str:
        """Extract and format context from ChromaDB results"""
        if not results or not results.get("metadatas"):
            return ""
            
        context_parts = []
        for metadata in results["metadatas"][0]:
            if metadata and "script" in metadata:
                context_parts.append(metadata["script"])
        return "\n\n".join(context_parts)

    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI with proper context handling"""
        prompt = f"Context:\n{context}\n\nQuery: {query}" if context else query
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Answer based on the context below. If unsure, say so.\n\n{prompt}"
            }],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content