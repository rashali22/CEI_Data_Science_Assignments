"""
Logging Utils Module (Phase 7)

Provides functions to record Q&A interactions, track pipeline settings,
and export session logs as JSON or CSV format.
"""

import json
import csv
import io
from datetime import datetime
from typing import List, Dict, Any, Optional


def create_log_record(
    question: str,
    answer: str,
    is_fallback: bool,
    top_score: float,
    retrieved_chunks: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Creates a structured log record for a Q&A interaction.

    Args:
        question: User query text.
        answer: Generated answer or fallback text.
        is_fallback: Boolean indicating if threshold fallback occurred.
        top_score: Similarity score of top retrieved chunk.
        retrieved_chunks: List of retrieved chunk dictionaries.
        settings: Pipeline configuration dictionary.

    Returns:
        Structured log record dictionary.
    """
    default_settings = {
        "top_k": 4,
        "confidence_threshold": 0.35,
        "hybrid_search": False,
        "reranking": False,
        "chunk_size_preset": "Medium",
        "embedding_model": "all-MiniLM-L6-v2"
    }

    if settings:
        default_settings.update(settings)

    chunk_summaries = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "source": c.get("source", ""),
            "page": c.get("page"),
            "score": c.get("score", 0.0)
        }
        for c in retrieved_chunks
    ]

    return {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "is_fallback": is_fallback,
        "top_score": top_score,
        "retrieved_chunks": chunk_summaries,
        "settings": default_settings
    }


def export_logs_json(log_records: List[Dict[str, Any]]) -> str:
    """
    Serializes log records into a formatted JSON string.
    """
    return json.dumps(log_records, indent=2)


def export_logs_csv(log_records: List[Dict[str, Any]]) -> str:
    """
    Exports log records into a CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "timestamp",
        "question",
        "answer",
        "is_fallback",
        "top_score",
        "retrieved_chunk_ids",
        "retrieved_scores",
        "top_k",
        "confidence_threshold",
        "hybrid_search",
        "reranking",
        "chunk_size_preset",
        "embedding_model"
    ])

    for record in log_records:
        chunks = record.get("retrieved_chunks", [])
        chunk_ids = ";".join([c.get("chunk_id", "") for c in chunks])
        chunk_scores = ";".join([str(c.get("score", 0.0)) for c in chunks])
        st = record.get("settings", {})

        writer.writerow([
            record.get("timestamp", ""),
            record.get("question", ""),
            record.get("answer", ""),
            record.get("is_fallback", False),
            record.get("top_score", 0.0),
            chunk_ids,
            chunk_scores,
            st.get("top_k", 4),
            st.get("confidence_threshold", 0.35),
            st.get("hybrid_search", False),
            st.get("reranking", False),
            st.get("chunk_size_preset", "Medium"),
            st.get("embedding_model", "all-MiniLM-L6-v2")
        ])

    return output.getvalue()
