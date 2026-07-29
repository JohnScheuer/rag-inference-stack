from pydantic import BaseModel
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]

class IngestResponse(BaseModel):
    status: str
    chunks_added: int
