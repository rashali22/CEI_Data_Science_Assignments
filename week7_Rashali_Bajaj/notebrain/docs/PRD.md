# NoteBrain QA — Product Requirements Document (PRD)

## 1. Project Title
NoteBrain QA

## 2. Overview & Objective
NoteBrain QA is a local, privacy-focused Streamlit RAG (Retrieval-Augmented Generation) application designed for querying personal and study notes (PDFs and TXT documents). It enables fast document ingestion, vector retrieval, grounded answer generation, and experimental toggles for evaluation.

## 3. Core Capabilities
- **Multi-format Ingestion**: Parse PDF (page-aware via `pypdf`) and TXT files.
- **Configurable Chunking**: Overlapping text chunking with full metadata tracking (source file, page number).
- **Persistent Vector Storage**: Store and manage document embeddings locally using ChromaDB.
- **Vector & Hybrid Search**: Perform dense similarity search with optional BM25 keyword blending.
- **Reranking**: Option to rerank top candidates using cross-encoder models (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Grounded LLM Generation**: Produce accurate, context-bound answers using Groq API (`llama-3.1-8b-instant`).
- **Confidence Fallback**: Refuse to answer if top retrieval similarity falls below a strict threshold (default 0.35).
- **Source Citations**: Display expandable transparent source attribution for every answer.
- **Document Management**: Remove indexed documents and their embeddings cleanly.
- **Q&A Logging & Evaluation**: Log queries, retrieved context, settings, and outputs for export and performance metrics analysis.

## 4. Architecture & Module Structure
- `app.py`: Streamlit entry point, UI layout, user interaction state.
- `modules/ingestion.py`: Text extraction for PDF and TXT files.
- `modules/chunking.py`: Text splitting into overlapping chunks with metadata.
- `modules/embeddings.py`: SentenceTransformer embedding wrappers (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`).
- `modules/vectorstore.py`: ChromaDB client, collection management, document addition & deletion.
- `modules/retrieval.py`: Dense vector retrieval, hybrid search (BM25 blend), cross-encoder re-ranking.
- `modules/generation.py`: Groq LLM client, prompt templates, grounded Q&A generation.
- `modules/logging_utils.py`: Event logging, session history tracking, CSV/JSON export.

## 5. Technology Stack
- **UI Framework**: Streamlit
- **PDF Extraction**: `pypdf`
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Store**: `chromadb`
- **Keyword Search**: `rank_bm25`
- **LLM Provider**: `groq` (Groq API, model `llama-3.1-8b-instant`)
- **Environment Management**: `python-dotenv`

## 6. System Requirements & Rules
1. Only implement features strictly scoped to the active phase.
2. All user interactions must run smoothly in Streamlit without unhandled exceptions.
3. Keep code modular, clean, and maintainable.
