"""
Retrieval Module (Phase 3, Phase 8, Phase 9, & Phase 11)

Handles query embedding, dense vector similarity search, BM25 hybrid search,
and local CrossEncoder re-ranking. Supports model selection (all-MiniLM-L6-v2, all-mpnet-base-v2).
"""

import math
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from modules.embeddings import embed_texts
from modules.vectorstore import get_collection, COLLECTION_NAME

# Global CrossEncoder model cache
_CROSS_ENCODER_CACHE = {}


def get_cross_encoder_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    Loads and caches the CrossEncoder model.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


def sigmoid(x: float) -> float:
    """
    Applies sigmoid activation to normalize cross-encoder logits into [0, 1] range.
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def retrieve_chunks(
    query: str,
    top_k: int = 4,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
    collection_name: str = COLLECTION_NAME
) -> List[Dict[str, Any]]:
    """
    Retrieves top-k chunks from ChromaDB with optional BM25 hybrid search and CrossEncoder re-ranking.
    """
    query = query.strip()
    if not query:
        return []

    collection = get_collection(collection_name)
    total_count = collection.count()
    if total_count == 0:
        return []

    candidate_k = min(max(top_k * 3, 12), total_count) if (use_hybrid or use_rerank) else min(top_k, total_count)

    # Embed query string using selected embedding model
    query_vectors = embed_texts([query], model_name=embedding_model)
    if not query_vectors:
        return []

    try:
        results = collection.query(
            query_embeddings=query_vectors,
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        if "dimension" in str(e).lower():
            # If dimension mismatch occurs against collection, fallback to the alternate model
            alt_model = "all-mpnet-base-v2" if embedding_model != "all-mpnet-base-v2" else "all-MiniLM-L6-v2"
            query_vectors = embed_texts([query], model_name=alt_model)
            results = collection.query(
                query_embeddings=query_vectors,
                n_results=candidate_k,
                include=["documents", "metadatas", "distances"]
            )
        else:
            raise e

    if not results or "ids" not in results or not results["ids"] or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    candidates: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        c_id = ids[i]
        text = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else 0.0

        source = meta.get("source", "Unknown") if meta else "Unknown"
        raw_page = meta.get("page", -1) if meta else -1
        page = raw_page if raw_page != -1 else None

        vec_score = max(0.0, 1.0 - dist)
        candidates.append({
            "chunk_id": c_id,
            "source": source,
            "page": page,
            "text": text,
            "score": round(vec_score, 4),
            "distance": round(dist, 4)
        })

    # Step 1: Optional Hybrid Search (BM25 Blend)
    if use_hybrid:
        corpus = [c["text"].lower().split() for c in candidates]
        bm25 = BM25Okapi(corpus)
        tokenized_query = query.lower().split()
        bm25_raw_scores = bm25.get_scores(tokenized_query)

        max_bm25 = max(bm25_raw_scores) if len(bm25_raw_scores) > 0 and max(bm25_raw_scores) > 0 else 1.0

        for i, c in enumerate(candidates):
            norm_bm25 = bm25_raw_scores[i] / max_bm25 if max_bm25 > 0 else 0.0
            c["score"] = round(0.5 * c["score"] + 0.5 * norm_bm25, 4)

        candidates.sort(key=lambda x: x["score"], reverse=True)

    # Step 2: Optional CrossEncoder Re-ranking
    if use_rerank and candidates:
        ce_model = get_cross_encoder_model("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, c["text"]] for c in candidates]
        raw_ce_scores = ce_model.predict(pairs)

        for i, c in enumerate(candidates):
            raw_score = float(raw_ce_scores[i])
            normalized_ce_score = round(sigmoid(raw_score), 4)
            c["score"] = normalized_ce_score

        candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates[:top_k]
