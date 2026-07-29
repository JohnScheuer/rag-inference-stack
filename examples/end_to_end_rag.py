from src.retrieval.store import VectorStore
from src.embeddings.models import EmbeddingEngine
from src.generation.llm_client import LocalLLMClient
from src.generation.rag_pipeline import RAGPipeline
from src.utils.config import config

# Mock do objeto Retriever para o pipeline
class SimpleRetriever:
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

# 1. Instanciar componentes
embedder = EmbeddingEngine(config.embedding_model_name)
store = VectorStore(config.chroma_persist_dir)
llm = LocalLLMClient(config.llm_model_name)
retriever = SimpleRetriever(embedder, store)

# 2. Setup Pipeline
pipeline = RAGPipeline(retriever, llm)

# 3. Query
question = "Explain how speculative decoding works based on the provided text."
print(f"\nQuestion: {question}\n")

result = pipeline.answer(question)

print("=" * 50)
print(f"RAG ANSWER:\n{result['answer']}")
print("=" * 50)
print(f"Sources: {[s['source'] for s in result['sources']]}")
