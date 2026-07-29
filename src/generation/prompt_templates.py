RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

def build_rag_prompt(question: str, context_chunks: list) -> str:
    context_text = "\n\n".join(context_chunks)
    return RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)
