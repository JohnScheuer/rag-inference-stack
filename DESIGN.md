# DESIGN.md — rag-inference-stack

## Architecture Overview

The **rag-inference-stack** is a modular Retrieval-Augmented Generation (RAG) system 
designed for local-first deployment. It decouples document ingestion from 
real-time inference.

### Data Flow

1. **Ingestion Path:**
   - Raw Documents (PDF/TXT) → `RAGIngestor` → Text Chunks.
   - Text Chunks → `EmbeddingEngine` → Vector Representations.
   - Vectors + Metadata → `VectorStore` (ChromaDB).

2. **Inference Path:**
   - User Query → `EmbeddingEngine` → Query Vector.
   - Query Vector → `VectorStore` → Top-K Relevant Context.
   - Context + Query → `RAGPipeline` → Prompt Template.
   - Prompt → `LocalLLMClient` (Qwen2) → Generated Answer.

---

## Technical Stack & Decisions

- **Framework:** FastAPI (High-performance asynchronous API layer).
- **Vector Database:** ChromaDB (Local, persistent, zero-dependency).
- **Embeddings:** `all-MiniLM-L6-v2` (Excellent performance-to-size ratio).
- **LLM:** Qwen2-0.5B-Instruct (Optimized for 8GB VRAM environments).
- **Precision:** FP16 (To maximize GPU throughput on RTX 2070).

---

## Performance Analysis

Based on end-to-end benchmarks:

- **Median Latency:** ~1820ms per query.
- **Throughput Bottleneck:** LLM Generation (Autoregressive sampling).
- **Retrieval Efficiency:** Vector search in ChromaDB accounts for <1% of total latency.
- **Memory Footprint:** 
  - Embedding Model: ~400MB.
  - LLM Model: ~1.1GB.
  - Overhead/VDB: ~200MB.
  - **Total Peak VRAM:** ~1.7GB (Ideal for consumer GPUs).

---

## Future Roadmap

- **Hybrid Retrieval:** Integrating BM25 (keyword search) with dense similarity.
- **Reranking:** Adding a Cross-Encoder stage to improve retrieval precision.
- **Chunking Strategies:** Implementing semantic chunking based on sentence embeddings.
- **Evaluation:** Automating faithfulness and relevance scoring (RAGAS pattern).
