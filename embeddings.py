# backend/embeddings.py

from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, TOP_K_INITIAL, TOP_K_RERANK

# Load HuggingFace sentence-transformers model (runs locally, free)
_embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# In-memory vector store
# { paper_id: [ { "embedding": np.array, "text": str, "chunk_id": int } ] }
VECTOR_STORE: Dict[str, List[Dict]] = {}


def get_embedding(texts: List[str]) -> np.ndarray:
    """
    Returns numpy array of embeddings, shape (len(texts), dim).
    Uses local SentenceTransformer model (no API).
    """
    # normalize_embeddings=True gives cosine-ready vectors
    embs = _embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embs


def index_paper(paper_id: str, chunks: List[Dict]):
    """
    Compute embeddings for all chunks and store them in VECTOR_STORE.
    """
    global VECTOR_STORE
    texts = [c["text"] for c in chunks]
    embeddings = get_embedding(texts)

    records = []
    for c, emb in zip(chunks, embeddings):
        records.append({
            "chunk_id": c["id"],
            "text": c["text"],
            "embedding": emb,
        })

    VECTOR_STORE[paper_id] = records


def _keyword_overlap_score(query: str, doc: str) -> float:
    """
    Very simple lexical score:
    ratio of overlapping unique tokens (lowercased, alnum).
    """
    tokenize = lambda s: set(re.findall(r"\b\w+\b", s.lower()))
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    if not q_tokens or not d_tokens:
        return 0.0
    overlap = q_tokens & d_tokens
    return len(overlap) / len(q_tokens)


def search_with_rerank(paper_id: str, query: str) -> List[Dict]:
    """
    2-stage retrieval:
      1. Dense retrieval with cosine similarity (HF embeddings).
      2. Re-ranking using hybrid score:
         final_score = 0.7 * cosine_sim + 0.3 * keyword_overlap.
    Returns top-k reranked chunks (list of dicts).
    """
    if paper_id not in VECTOR_STORE or not VECTOR_STORE[paper_id]:
        return []

    records = VECTOR_STORE[paper_id]

    # --- Stage 1: dense retrieval ---
    query_emb = get_embedding([query])[0].reshape(1, -1)
    doc_embs = np.vstack([r["embedding"] for r in records])
    cos_scores = cosine_similarity(query_emb, doc_embs)[0]  # shape: (num_docs,)

    # attach initial scores
    for r, s in zip(records, cos_scores):
        r["cosine_score"] = float(s)

    # pick top-K by cosine similarity
    top_indices = np.argsort(-cos_scores)[:TOP_K_INITIAL]
    initial_candidates = [records[i] for i in top_indices]

    # --- Stage 2: re-ranking (hybrid) ---
    reranked = []
    for r in initial_candidates:
        lex_score = _keyword_overlap_score(query, r["text"])
        hybrid = 0.7 * r["cosine_score"] + 0.3 * lex_score
        reranked.append({
            **r,
            "lex_score": lex_score,
            "hybrid_score": hybrid,
        })

    reranked_sorted = sorted(reranked, key=lambda x: x["hybrid_score"], reverse=True)
    return reranked_sorted[:TOP_K_RERANK]
