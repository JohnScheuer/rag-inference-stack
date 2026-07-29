from src.generation.prompt_templates import build_rag_prompt

class RAGPipeline:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(self, question: str, top_k: int = 3) -> dict:
        # 1. Recuperar contexto
        query_vec = self.retriever.embedding_model.encode([question])[0]
        results = self.retriever.vector_store.query(query_vec, top_k=top_k)
        context_chunks = results['documents'][0]
        
        # 2. Construir Prompt
        prompt = build_rag_prompt(question, context_chunks)
        
        # 3. Gerar Resposta
        answer = self.llm_client.generate(prompt)
        
        return {
            "answer": answer,
            "sources": results['metadatas'][0]
        }
