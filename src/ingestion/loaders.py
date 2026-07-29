from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader

class Document:
    def __init__(self, content: str, metadata: Dict):
        self.content = content
        self.metadata = metadata

class RAGIngestor:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_file(self, path: str) -> List[Document]:
        p = Path(path)
        if p.suffix == ".pdf":
            reader = PdfReader(p)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
        return self._chunk_text(text, str(p))

    def _chunk_text(self, text: str, source: str) -> List[Document]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(Document(
                content=chunk, 
                metadata={"source": source, "start": start, "end": end}
            ))
            start += self.chunk_size - self.overlap
        return chunks
