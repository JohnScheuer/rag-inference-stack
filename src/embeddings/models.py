from sentence_transformers import SentenceTransformer
import torch
from typing import List

class EmbeddingEngine:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
