"""
Generation Module (Phase 4)

Handles grounded answer generation using Groq API (llama-3.1-8b-instant)
and confidence threshold checks to prevent hallucinated answers.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

DEFAULT_CONFIDENCE_THRESHOLD = 0.35
FALLBACK_MESSAGE = "I couldn't find anything relevant to this in your uploaded documents."
DEFAULT_MODEL = "llama-3.1-8b-instant"


def get_groq_client() -> Groq:
    """
    Initializes and returns Groq client using GROQ_API_KEY from environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not configured in .env file.")
    return Groq(api_key=api_key)


def build_grounded_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Assembles prompt context clearly labeled with source and page metadata per PRD Section 13.
    """
    context_blocks = []
    for idx, c in enumerate(retrieved_chunks, start=1):
        source = c.get("source", "Unknown")
        page = c.get("page")
        page_str = f"Page {page}" if page is not None else "Page N/A"
        text = c.get("text", "").strip()

        context_blocks.append(f"[Source: {source}, {page_str}]\n{text}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""Context:
{context_str}

Instructions:
Answer the user's question concisely using ONLY the provided context above. If the context does not contain sufficient information to answer the question, state that you cannot answer based on the provided context. Do not invent or extrapolate details outside the provided context.

Question: {query}
Answer:"""

    return prompt


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_name: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Generates a grounded answer or returns fallback message if top similarity score is below threshold.

    Args:
        query: User question string.
        retrieved_chunks: List of retrieved chunk dicts from retrieval.py.
        confidence_threshold: Minimum similarity score required to trigger LLM call.
        model_name: Groq LLM model name.

    Returns:
        Dict:
        {
            "answer": str,
            "is_fallback": bool,
            "top_score": float
        }
    """
    if not retrieved_chunks:
        return {
            "answer": FALLBACK_MESSAGE,
            "is_fallback": True,
            "top_score": 0.0
        }

    top_score = retrieved_chunks[0].get("score", 0.0)

    # Confidence threshold fallback check (PRD Section 8.1)
    if top_score < confidence_threshold:
        return {
            "answer": FALLBACK_MESSAGE,
            "is_fallback": True,
            "top_score": top_score
        }

    # Generate answer via Groq API
    try:
        client = get_groq_client()
        prompt_content = build_grounded_prompt(query, retrieved_chunks)

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are NoteBrain QA, a helpful assistant that provides accurate, grounded answers strictly based on the user's notes and documents."
                },
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            temperature=0.2,
            max_tokens=600
        )

        answer_text = completion.choices[0].message.content.strip()
        return {
            "answer": answer_text,
            "is_fallback": False,
            "top_score": top_score
        }

    except Exception as e:
        error_msg = f"Error during answer generation: {str(e)}"
        return {
            "answer": f"⚠️ {error_msg}",
            "is_fallback": True,
            "top_score": top_score
        }
