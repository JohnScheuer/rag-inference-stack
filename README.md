<div align="center">

# rag-inference-stack

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%20+-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB]](https://img.shields.io/badge/VectorDB-Chroma-yellow.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Retrieval-Augmented Generation stack with document ingestion, vector retrieval, and LLM generation.**

Median Latency: 1.8s | Local-first design | Persistent Vector Store

</div>

---

## Features

- **End-to-End RAG:** Ingestion, chunking, embedding, retrieval, and generation.
- **Persistent Storage:** Local ChromaDB instance saves indexed documents across sessions.
- **REST API:** FastAPI endpoints for asynchronous document ingestion and querying.
- **Optimized for Consumer GPU:** Low VRAM footprint (~1.7GB), ideal for 8GB cards.
- **Context Citations:** Returns sources used for every generated answer.

---

## Performance (RTX 2070 8GB)

Benchmarked with Qwen2-0.5B and all-MiniLM-L6-v2.

| Metric | Value |
| :--- | :--- |
| **Median End-to-End Latency** | **1820.40 ms** |
| Min Latency | 1633.04 ms |
| Peak VRAM Usage | ~1.7 GB |
| Retrieval Accuracy (Recall@1) | 1.0 (on tested corpus) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
export PYTHONPATH=.
uvicorn src.server.main:app --port 8000

# 3. Ingest a document (in another terminal)
curl -X POST http://localhost:8000/ingest \
     -F "file=@data/sample_docs/test.txt"

# 4. Ask a question
curl -X POST http://localhost:8000/query \
     -H" Content-Type: application/json" \
     -d '{"question": "What is speculative decoding?", "top_k": 3}'`
```

---

## Documentation

- [DESIGN.md](DESIGN.md): Architecture and technical decisions.
- [summary.txt](summary.txt): Concise project report and metrics.

---

## Related Projects

- [lora-inference-runtime](https://github.com/JohnScheuer/lora-inference-runtime): Multi-adapter LoRA serving.
- [speculative-decoding-runtime](https://github.com/JohnScheuer/speculative-decoding-runtime): Acceleration via draft-model verification.
- [vlm-inference/runtime]https://github.com/JohnScheuer/vlm-inference-runtime): Multi-modal deployment.

---

## License

[MIT](LICENSE) - Joao Felipe De Souza, 2026
