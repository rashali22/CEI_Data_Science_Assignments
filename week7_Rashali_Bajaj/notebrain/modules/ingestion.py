"""
Ingestion Module (Phase 1)

Handles document text extraction for PDF and TXT files, preserving page numbers for PDFs.
"""

import io
from pypdf import PdfReader
from typing import List, Dict, Any


def extract_text_from_file(uploaded_file: Any) -> List[Dict[str, Any]]:
    """
    Extracts text from an uploaded file (PDF or TXT).

    Args:
        uploaded_file: Streamlit UploadedFile or file-like object / file path.

    Returns:
        List of dicts: [{'page': int or None, 'text': str}]
    """
    filename = getattr(uploaded_file, "name", "")
    pages_data: List[Dict[str, Any]] = []

    if filename.lower().endswith(".pdf"):
        if hasattr(uploaded_file, "read"):
            content = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            reader = PdfReader(io.BytesIO(content))
        else:
            reader = PdfReader(uploaded_file)

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            extracted = page.extract_text() or ""
            pages_data.append({
                "page": page_num,
                "text": extracted
            })
    else:
        # Treat as TXT file
        if hasattr(uploaded_file, "read"):
            raw_bytes = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            extracted = raw_bytes.decode("utf-8", errors="replace")
        elif isinstance(uploaded_file, str):
            with open(uploaded_file, "r", encoding="utf-8", errors="replace") as f:
                extracted = f.read()
        else:
            extracted = str(uploaded_file)

        pages_data.append({
            "page": None,
            "text": extracted
        })

    return pages_data
