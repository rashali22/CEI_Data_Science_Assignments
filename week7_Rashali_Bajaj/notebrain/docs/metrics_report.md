# NoteBrain QA — Metrics & System Evaluation Report

**Project Title:** NoteBrain QA — Local Streamlit RAG Application  
**PRD Version:** 1.0 (Section 14.3 Deliverable)  
**Date:** August 2026  

---

## 1. System Setup & Pipeline Architecture

### 1.1 Document Ingestion & Chunking
- **Supported Formats:** PDF (page-aware via `pypdf`) and TXT files.
- **Chunk Size Presets Tested:**
  - **Small:** 250 characters, 25 character overlap (~50 words)
  - **Medium:** 500 characters, 50 character overlap (~100 words, Default)
  - **Large:** 1000 characters, 100 character overlap (~200 words)
- **Metadata Preserved:** Source filename, page number (1-indexed for PDF, `N/A` for TXT), unique chunk ID.

### 1.2 Embedding Models & Vector Store
- **Embedding Models Supported:**
  - `all-MiniLM-L6-v2`: 384 dimensions (Default, fast local inference)
  - `all-mpnet-base-v2`: 768 dimensions (Higher semantic accuracy)
- **Vector Database:** ChromaDB persistent store (`chromadb.PersistentClient`) saved locally in `data/chroma_db/`.
- **Similarity Metric:** Cosine similarity ($s = 1 - d$, where $d$ is ChromaDB distance).

### 1.3 Retrieval Strategies
- **Dense Vector Search:** Nearest-neighbor vector similarity retrieval ($k=4$).
- **Hybrid Search Toggle:** Blends 50% vector similarity score with 50% BM25 keyword matching score (`rank_bm25`).
- **Re-ranking Toggle:** Local CrossEncoder re-scoring using `cross-encoder/ms-marco-MiniLM-L-6-v2` over candidate chunks.

### 1.4 LLM Generation & Confidence Threshold
- **LLM Provider & Model:** Groq API (`llama-3.1-8b-instant`).
- **Prompt Strategy:** Strictly grounded context injection with explicit `[Source: filename, page N]` tags and instructions to refuse answering if context is insufficient.
- **Confidence Fallback Cutoff:** Threshold = `0.35` similarity score. If top chunk similarity < `0.35`, the LLM call is bypassed and the fixed fallback message is returned:
  *"I couldn't find anything relevant to this in your uploaded documents."*

---

## 2. Evaluation Flow & Output Logging

- **Evaluation Questions File:** `eval/eval_questions.json` (5-10 structured benchmark questions).
- **Execution Workflow:** Interactive evaluation trigger in `app.py` Advanced Settings or standalone execution via `python eval/run_eval.py`.
- **Validation Logs:** Saved automatically to `eval/logs/` in both timestamped `.json` and `.csv` formats.

---

## 3. Qualitative Comparison & Evaluation Notes

*Fill in your personal qualitative observations below after running evaluations across different pipeline settings:*

- **Hybrid search (BM25 + Vector) vs. Pure vector search:**  
  `____________________________________________________________________________________________________`

- **Cross-encoder re-ranking vs. Raw vector retrieval:**  
  `____________________________________________________________________________________________________`

- **Chunk size impact (Small [250] vs. Medium [500] vs. Large [1000]):**  
  `____________________________________________________________________________________________________`

- **Embedding model performance (`all-MiniLM-L6-v2` [384d] vs. `all-mpnet-base-v2` [768d]):**  
  `____________________________________________________________________________________________________`

- **Hallucination suppression & confidence threshold fallback accuracy (0.35 cutoff):**  
  `____________________________________________________________________________________________________`

---

## 4. Final System Verification Checklist
- [x] Document ingestion & page-aware chunking
- [x] ChromaDB persistent vector index & sidebar management
- [x] Top-k vector retrieval
- [x] Grounded answer generation via Groq API
- [x] Confidence threshold fallback
- [x] Expandable source citations UI (`filename, page N`)
- [x] Document deletion functionality
- [x] Session Q&A logging & CSV/JSON export
- [x] Hybrid search experiment toggle (BM25)
- [x] Re-ranking experiment toggle (CrossEncoder)
- [x] Chunk size preset control & safe re-indexing
- [x] Embedding model toggle & dimension mismatch protection
- [x] Evaluation runner & system metrics report deliverable
