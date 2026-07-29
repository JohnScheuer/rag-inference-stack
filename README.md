<div align="center">

# rag-inference-stack

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-blue)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Retrieval-Augmented Generation stack with document ingestion, vector retrieval, and LLM generation.**

Median Latency: 1.8s | Local-first design | 8GB GPU compatible

</div>

---

## What This Does

Retrieval-Augmented Generation (RAG) combines document retrieval with LLM generation. Instead of relying solely on LLM training data, RAG retrieves relevant documents and includes them as context for grounded answers.

This project implements the full RAG pipeline:
1. Document ingestion (PDF, text, markdown)
2. Chunking (fixed-size and sentence-based)
3. Embedding generation (sentence-transformers)
4. Vector retrieval (ChromaDB)
5. LLM generation with retrieved context
6. FastAPI HTTP layer with ingestion and query endpoints

---

## Performance (RTX 2070 8GB)

Benchmarked with Qwen2-0.5B (LLM) and all-MiniLM-L6-v2 (embeddings).

| Metric | Value |
| :--- | :--- |
| **Median End-to-End Latency** | **1820 ms** |
| Min Latency | 1633 ms |
| Peak VRAM Usage | ~1.7 GB |
| Retrieval Recall@1 | 1.0 (small tested corpus) |

**Latency breakdown (approximate):**
- Query embedding: ~5ms
- Vector search: ~10ms
- Context building: ~5ms
- LLM generation (200 tokens): ~1800ms

*The LLM generation dominates end-to-end latency. Retrieval overhead is negligible.*

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start server
export PYTHONPATH=.
uvicorn src.server.main:app --port 8000

# 3. Ingest a document (another terminal)
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample_docs/test.txt"

# 4. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '`{"question": "What is speculative decoding?", "top_k": 3}`'
```

---

## Design Decisions

- **Why ChromaDB?** Local persistent storage, no external dependencies. Good for MVP demonstration.
- **Why sentence-transformers?** Established baseline embeddings. Fine-tuned models would improve domain-specific quality.
- **Why local LLM?** Self-contained deployment, fits in 8GB VRAM. LLMClient interface allows swapping for OpenAI/Anthropic APIs.

---

## Limitations

- Single-node ChromaDB (no distributed retrieval).
- Basic sentence-transformer embeddings (no fine-tuning).
- Simple recall@K evaluation on small corpus (not production-grade).
- Sequential query processing (no batching).
- No reranking layer (relies solely on embedding similarity).

---

## Related Projects

Part of my LLM inference systems portfolio:
- [lora-inference-runtime](https://github.com/JohnScheuer/lora-inference-runtime): Multi-adapter LoRA serving.
- [speculative-decoding-runtime](https://github.com/JohnScheuer/speculative-decoding-runtime): Acceleration via draft-model verification.
- [vlm-inference-runtime](https://github.com/JohnScheuer/vlm-inference-runtime): Multimodal deployment.

---

## License

[MIT](LICENSE) - Joao Felipe De Souza, 2026
