# backend/rag.py

from typing import List, Dict
import os

from groq import Groq

from .embeddings import search_with_rerank
from .config import LLM_MODEL

# Groq client – uses GROQ_API_KEY from environment or .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_context(chunks: List[Dict]) -> str:
    """
    Concatenate chunks with markers for the LLM.
    """
    parts = []
    for c in chunks:
        parts.append(f"[CHUNK ID: {c['chunk_id']}]\n{c['text']}\n")
    return "\n\n".join(parts)


def answer_question(paper_id: str, question: str) -> Dict:
    """
    RAG pipeline with re-ranking + Groq LLM.
    Returns dict with answer + used_chunks.
    """
    retrieved_chunks = search_with_rerank(paper_id, question)
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find any relevant content for this paper.",
            "chunks": [],
        }

    context = build_context(retrieved_chunks)

    system_prompt = (
        "You are an AI assistant helping students understand research papers.\n"
        "Use only the provided context to answer the user's question.\n"
        "If the answer is not in the context, say you cannot find it in the paper.\n"
        "Explain in simple words. At the end, list the CHUNK IDs you used."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
    )

    answer = resp.choices[0].message.content

    return {
        "answer": answer,
        "chunks": retrieved_chunks,
    }
