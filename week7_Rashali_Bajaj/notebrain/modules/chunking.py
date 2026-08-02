"""
Chunking Module (Phase 1 & Phase 10)

Splits extracted text into overlapping chunks with configurable chunk size and overlap,
preserving source metadata (filename, page number).
Provides preset definitions for Small, Medium, and Large chunk sizes.
"""

from typing import List, Dict, Any, Tuple

# Presets defined per PRD Section 9.6 & 16
CHUNK_PRESETS: Dict[str, Tuple[int, int]] = {
    "Small": (250, 25),
    "Medium": (500, 50),
    "Large": (1000, 100)
}


def chunk_text_block(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Splits a single block of text into character-level chunks with overlap.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - chunk_overlap)
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)
        if end >= text_len:
            break
        start += step

    return chunks


def chunk_document(
    pages_data: List[Dict[str, Any]],
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits document pages data into structured chunk dictionaries with source and page metadata.
    """
    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for item in pages_data:
        page_num = item.get("page")
        page_text = item.get("text", "")

        text_chunks = chunk_text_block(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for t_chunk in text_chunks:
            chunk_counter += 1
            page_tag = f"p{page_num}" if page_num is not None else "na"
            all_chunks.append({
                "chunk_id": f"{filename}_{page_tag}_{chunk_counter}",
                "source": filename,
                "page": page_num,
                "text": t_chunk
            })

    return all_chunks
