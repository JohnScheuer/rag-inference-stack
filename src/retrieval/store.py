import chromadb
from typing import List, List
from src.ingestion.loaders import Document

class VectorStore:
    def __init__(self, persist_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection("rag_docs")

    def add(self, documents: List[Document], embeddings: List[List[float]]):
        ids = [f"id_{i}_{hash(d.content)}" for i, d in enumerate(documents)]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[d.content for d in documents],
            metadatas=[d.metadata for d in documents]
        )

    def query(self, query_embedding: List[float], top_k: int = 3):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
