from fastapi import FastAPI, UploadFile, File, Request
from contextlib import asynccontextmanager
import os
import shutil
from pathlib import Path

from src.utils.config import config
from src.ingestion.loaders import RAGIngestor
from src.embeddings.models import EmbeddingEngine
from src.retrieval.store import VectorStore
from src.generation.llm_client import LocalLLMClient
from src.generation.rag_pipeline import RAGPipeline
from src.server.schemas import QueryRequest, QueryResponse, IngestResponse

# Mock para manter compatibilidade com o pipeline
class SimpleRetriever:
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

# Estado global para os modelos
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Carrega modelos
    print("Iniciando RAG Stack...")
    embedder = EmbeddingEngine(config.embedding_model_name)
    store = VectorStore(config.chroma_persist_dir)
    llm = LocalLLMClient(config.llm_model_name)
    retriever = SimpleRetriever(embedder, store)
    
    state["pipeline"] = RAGPipeline(retriever, llm)
    state["ingestor"] = RAGIngestor(config.chunk_size, config.chunk_overlap)
    state["embedder"] = embedder
    state["store"] = store
    
    print("Servidor RAG pronto!")
    yield
    # Shutdown: Limpeza se necessário
    state.clear()

app = FastAPI(lifespan=lifespan, title="RAG Inference Stack API")

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    result = state["pipeline"].answer(request.question, top_k=request.top_k)
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    # Salva arquivo temporário
    temp_path = Path(f"data/sample_docs/{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Processa
    chunks = state["ingestor"].load_file(str(temp_path))
    embeddings = state["embedder"].encode([c.content for c in chunks])
    state["store"].add(chunks, embeddings)
    
    return {"status": "success", "chunks_added": len(chunks)}
