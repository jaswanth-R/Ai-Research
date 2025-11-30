# backend/config.py

# HuggingFace Sentence-Transformers model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq LLM model
# Check Groq docs for latest models – this is a good default.
LLM_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

MAX_CONTEXT_CHUNKS = 6
TOP_K_INITIAL = 12
TOP_K_RERANK = 6
