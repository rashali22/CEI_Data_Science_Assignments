"""
Evaluation Runner Script (Phase 12)

Executes sample questions from eval/eval_questions.json against the RAG pipeline
and saves timestamped logs into eval/logs/.
"""

import sys
import os
import json
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.retrieval import retrieve_chunks
from modules.generation import generate_answer, DEFAULT_CONFIDENCE_THRESHOLD
from modules.logging_utils import create_log_record, export_logs_json, export_logs_csv
from modules.vectorstore import get_collection, COLLECTION_NAME


def auto_detect_model(collection_name: str = COLLECTION_NAME) -> str:
    """
    Detects whether the Chroma collection contains 768-dim (all-mpnet-base-v2)
    or 384-dim (all-MiniLM-L6-v2) vectors.
    """
    try:
        col = get_collection(collection_name)
        res = col.get(include=["embeddings"], limit=1)
        if res and "embeddings" in res and res["embeddings"] and len(res["embeddings"][0]) == 768:
            return "all-mpnet-base-v2"
    except Exception:
        pass
    return "all-MiniLM-L6-v2"


def run_evaluation(
    eval_questions_file: str = None,
    output_dir: str = None,
    top_k: int = 4,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    chunk_preset: str = "Medium",
    embedding_model: str = None
):
    if eval_questions_file is None:
        eval_questions_file = os.path.join(PROJECT_ROOT, "eval", "eval_questions.json")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "eval", "logs")

    if embedding_model is None:
        embedding_model = auto_detect_model()

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(eval_questions_file):
        print(f"Error: {eval_questions_file} not found.")
        return []

    with open(eval_questions_file, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    print(f"Loaded {len(questions_data)} evaluation questions.")
    print(f"Pipeline settings: top_k={top_k}, hybrid={use_hybrid}, rerank={use_rerank}, chunk={chunk_preset}, model={embedding_model}")

    eval_logs = []

    for idx, item in enumerate(questions_data, start=1):
        q_text = item.get("question", "")
        print(f"\n[{idx}/{len(questions_data)}] Query: {q_text}")

        retrieved = retrieve_chunks(
            query=q_text,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            embedding_model=embedding_model
        )

        result = generate_answer(
            query=q_text,
            retrieved_chunks=retrieved,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
        )

        print(f" Top Score: {result['top_score']} | Fallback: {result['is_fallback']}")
        print(f" Answer preview: {result['answer'][:80]}...")

        record = create_log_record(
            question=q_text,
            answer=result["answer"],
            is_fallback=result["is_fallback"],
            top_score=result["top_score"],
            retrieved_chunks=retrieved,
            settings={
                "top_k": top_k,
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "hybrid_search": use_hybrid,
                "reranking": use_rerank,
                "chunk_size_preset": chunk_preset,
                "embedding_model": embedding_model
            }
        )
        eval_logs.append(record)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"eval_run_{timestamp_str}.json")
    csv_path = os.path.join(output_dir, f"eval_run_{timestamp_str}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(export_logs_json(eval_logs))

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(export_logs_csv(eval_logs))

    print(f"\nEvaluation completed! Saved logs to:\n - {json_path}\n - {csv_path}")
    return eval_logs


if __name__ == "__main__":
    run_evaluation()
