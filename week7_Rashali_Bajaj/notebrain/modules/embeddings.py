"""
Embeddings Module (Phase 2 & Phase 11)

Wraps sentence-transformers to convert text chunks into vector embeddings.
Supported models (PRD Section 9.3):
- "all-MiniLM-L6-v2" (384 dimensions, default)
- "all-mpnet-base-v2" (768 dimensions)
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Mapping from friendly model names to HuggingFace model strings
EMBEDDING_MODELS: Dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
}

# Global model cache to avoid reloading models on every call
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads and caches the specified SentenceTransformer model.
    """
    hf_model_path = EMBEDDING_MODELS.get(model_name, model_name)
    if hf_model_path not in _MODEL_CACHE:
        _MODEL_CACHE[hf_model_path] = SentenceTransformer(hf_model_path)
    return _MODEL_CACHE[hf_model_path]


def embed_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> List[List[float]]:
    """
    Embeds a list of text strings into vector representations using the chosen model.

    Args:
        texts: List of text strings to embed.
        model_name: Friendly model name or HuggingFace path.

    Returns:
        List of float vectors.
    """
    if not texts:
        return []

    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()
