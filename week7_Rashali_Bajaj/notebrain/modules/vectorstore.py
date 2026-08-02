"""
Vectorstore Module (Phase 2, Phase 6, Phase 10, & Phase 11)

Wraps a persistent ChromaDB client for indexing, querying, deleting, and resetting document collections.
Data path: data/chroma_db/
"""

import os
from typing import List, Dict, Any
import chromadb

CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
COLLECTION_NAME = "notebrain_collection"

_CHROMA_CLIENT = None


def get_chroma_client():
    """
    Returns persistent ChromaDB client instance.
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _CHROMA_CLIENT


def get_collection(collection_name: str = COLLECTION_NAME):
    """
    Retrieves or creates the Chroma collection.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)


def clear_collection(collection_name: str = COLLECTION_NAME) -> None:
    """
    Clears all items in the Chroma collection for re-indexing.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    client.get_or_create_collection(name=collection_name)


def add_chunks_to_vectorstore(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection_name: str = COLLECTION_NAME
) -> None:
    """
    Adds document chunks and their vector embeddings to the Chroma collection.
    Handles dimension mismatches automatically by resetting the collection if embedding model dimensions changed.
    """
    if not chunks or not embeddings:
        return

    collection = get_collection(collection_name)

    ids = [str(c["chunk_id"]) for c in chunks]
    documents = [str(c["text"]) for c in chunks]
    metadatas = [
        {
            "source": str(c.get("source", "Unknown")),
            "page": int(c["page"]) if c.get("page") is not None else -1
        }
        for c in chunks
    ]

    try:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    except Exception as e:
        error_str = str(e).lower()
        if "dimension" in error_str or "expecting embedding" in error_str:
            # Dimension mismatch: clear old collection and upsert with new vector dimensionality
            clear_collection(collection_name)
            new_col = get_collection(collection_name)
            new_col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        else:
            raise e


def get_indexed_documents(collection_name: str = COLLECTION_NAME) -> Dict[str, int]:
    """
    Lists all currently indexed documents in ChromaDB with their chunk counts.
    """
    collection = get_collection(collection_name)
    results = collection.get(include=["metadatas"])

    doc_counts: Dict[str, int] = {}
    if results and "metadatas" in results and results["metadatas"]:
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                source = meta["source"]
                doc_counts[source] = doc_counts.get(source, 0) + 1

    return doc_counts


def delete_document_from_vectorstore(
    source_filename: str,
    collection_name: str = COLLECTION_NAME
) -> bool:
    """
    Deletes all chunks belonging to a given source filename from the Chroma collection.
    """
    if not source_filename:
        return False

    collection = get_collection(collection_name)
    collection.delete(where={"source": source_filename})
    return True
