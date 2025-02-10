import chromadb
from sentence_transformers import SentenceTransformer

# ✅ Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Connect to ChromaDB
DB_PATH = "/app/RAG/script_db"  # Adjust the path if needed
db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_collection("scripts")


# ✅ Function to retrieve similar scripts
def retrieve_scripts(query, n_results=3):
    """
    Searches ChromaDB for relevant scripts based on the user query.
    Returns the retrieved scripts as a formatted string.
    """
    query_embedding = embedding_model.encode(query).tolist()

    # ✅ Query ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    if not results or "metadatas" not in results or not results.get("metadatas"):
        print("⚠️ No metadata found in ChromaDB query results.")
        return ""

    # ✅ Extract script text safely
    retrieved_texts = []
    for metadata in results["metadatas"][0]:  # Iterate through metadata list
        if isinstance(metadata, dict) and "script" in metadata:
            retrieved_texts.append(metadata["script"])
        else:
            print(f"⚠️ Metadata missing 'script' key: {metadata}")

    # ✅ Return retrieved scripts as a formatted string
    return "\n\n".join(retrieved_texts) if retrieved_texts else ""


# ✅ Process request before sending to model
def process_request(request: dict) -> dict:
    """
    This function modifies the Open WebUI request by adding retrieved script context
    before sending it to the language model.
    """

    # ✅ Extract user message (last message in the chat)
    if "messages" not in request or not isinstance(request["messages"], list):
        print("⚠️ Invalid request format. Missing 'messages' key or not a list.")
        return request  # Return the request unchanged if it's invalid

    query = request["messages"][-1].get("content", "")

    # ✅ Retrieve relevant scripts
    retrieved_context = retrieve_scripts(query)

    # ✅ Inject retrieved scripts into the system prompt
    if retrieved_context:
        system_message = {
            "role": "system",
            "content": f"Use the following scripts as reference when generating your response:\n\n{retrieved_context}"
        }
        request["messages"].insert(0, system_message)  # Add at the beginning

    # ✅ Return the modified request (pipeline continues)
    return request