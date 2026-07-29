from pydantic_settings import BaseSettings

class RAGConfig(BaseSettings):
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    chunk_size: int = 500
    chunk_overlap: int = 50
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "documents"
    device: str = "cuda"

config = RAGConfig()
