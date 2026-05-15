from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("clip_knowledge")

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_knowledge(query):

    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    return results["documents"][0]


YOUR ACTUAL STACK SHOULD NOW LOOK LIKE
RAG
+
Filesystem Tools
+
Terminal Tools
+
OBS Tools
+
SQLite Tools
+
Validation Layer
+
Structured Outputs

That is already a SERIOUS system.