from src.ingestion.loaders import RAGIngestor
from src.embeddings.models import EmbeddingEngine
from src.retrieval.store import VectorStore
from src.utils.config import config

# 1. Ingestão
print("Processando documentos...")
ingestor = RAGIngestor(chunk_size=200, overlap=20)
chunks = ingestor.load_file("data/sample_docs/test.txt")

# 2. Embeddings
print("Gerando embeddings...")
embedder = EmbeddingEngine(config.embedding_model_name)
embeddings = embedder.encode([c.content for c in chunks])

# 3. Store
print("Salvando no Vector DB...")
store = VectorStore(config.chroma_persist_dir)
store.add(chunks, embeddings)

# 4. Query Test
query = "How much speedup does speculative decoding provide?"
print(f"\nQuery: {query}")

query_vec = embedder.encode([query])[0]
results = store.query(query_vec, top_k=1)

print("-" * 30)
print(f"Resultado recuperado: {results['documents'][0][0]}")
print("-" * 30)
