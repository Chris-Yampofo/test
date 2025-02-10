import openai
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from sentence_transformers import SentenceTransformer
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage

class Pipeline:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.openai_client = None

    async def on_startup(self):
        # Initialize embedding model and load documents
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Llama Index setup (using a folder where index data is stored)
        from llama_index import VectorStoreIndex, SimpleDirectoryReader
        
        # Load documents (can be in ./data or /llamaIndex depending on where your documents are)
        document_loader = SimpleDirectoryReader("/llamaIndex/data")  # Adjust the path as needed
        documents = document_loader.load_data()
        self.index = VectorStoreIndex.from_documents(documents)

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
            
            # Use Llama Index to retrieve context (use the query engine for the retrieval)
            query_engine = self.index.as_query_engine()
            results = query_engine.query(user_message)
            
            # Process results from Llama Index
            context = self._process_results(results)
            
            # Generate response with OpenAI
            return self._generate_response(user_message, context)
            
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def _process_results(self, results: dict) -> str:
        """Extract and format context from Llama Index results"""
        if not results or not results.get("response"):
            return ""
        
        context_parts = []
        for result in results["response"]:
            if "text" in result:
                context_parts.append(result["text"])
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