# backend/chunking.py

from typing import List, Dict
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Simple sliding-window chunking.
    Returns list of dicts: {"id": int, "text": str}
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
            })
            chunk_id += 1
        start += chunk_size - overlap

    return chunks
