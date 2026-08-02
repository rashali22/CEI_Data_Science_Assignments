import streamlit as st
from modules.ingestion import extract_text_from_file
from modules.chunking import chunk_document, CHUNK_PRESETS
from modules.embeddings import embed_texts
from modules.vectorstore import (
    add_chunks_to_vectorstore,
    get_indexed_documents,
    delete_document_from_vectorstore,
    clear_collection
)
from modules.retrieval import retrieve_chunks
from modules.generation import generate_answer, DEFAULT_CONFIDENCE_THRESHOLD
from modules.logging_utils import create_log_record, export_logs_json, export_logs_csv
from eval.run_eval import run_evaluation

st.set_page_config(
    page_title="NoteBrain QA",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NoteBrain QA")
st.caption("A local Streamlit RAG application for document QA")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_log" not in st.session_state:
    st.session_state.qa_log = []

if "raw_documents" not in st.session_state:
    st.session_state.raw_documents = {}

if "indexed_embedding_model" not in st.session_state:
    st.session_state.indexed_embedding_model = "all-MiniLM-L6-v2"

# Sidebar - Advanced Settings Expander (Phase 8, 9, 10, 11, & 12)
with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    st.markdown("### Embedding Model Experiment")
    selected_embedding_model = st.selectbox(
        "Embedding model",
        options=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0,
        help="all-MiniLM-L6-v2 (384-dim, fast) vs all-mpnet-base-v2 (768-dim, higher accuracy)."
    )

    st.markdown("### Chunking Experiments")
    chunk_preset = st.selectbox(
        "Chunk size preset",
        options=["Small", "Medium", "Large"],
        index=1,
        help="Small (~250 chars), Medium (~500 chars), Large (~1000 chars)."
    )

    chunk_size, chunk_overlap = CHUNK_PRESETS[chunk_preset]

    st.markdown("### Search Experiments")
    use_hybrid = st.checkbox(
        "Use hybrid search (keyword + vector)",
        value=False,
        help="Blends BM25 keyword matching with vector similarity retrieval."
    )
    use_rerank = st.checkbox(
        "Re-rank results",
        value=False,
        help="Re-ranks candidate chunks using cross-encoder/ms-marco-MiniLM-L-6-v2."
    )

    st.divider()

    # Re-index Button
    if st.button("🔄 Re-index all documents", use_container_width=True, help="Re-runs chunking and embedding for all documents using active preset settings."):
        if not st.session_state.raw_documents:
            st.warning("No documents in session cache to re-index. Please upload documents first.")
        else:
            with st.spinner(f"Re-indexing all documents using model '{selected_embedding_model}' and '{chunk_preset}' preset..."):
                clear_collection()
                for doc_name, pages_data in st.session_state.raw_documents.items():
                    new_chunks = chunk_document(pages_data, filename=doc_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    if new_chunks:
                        texts_to_embed = [c["text"] for c in new_chunks]
                        embeddings = embed_texts(texts_to_embed, model_name=selected_embedding_model)
                        add_chunks_to_vectorstore(new_chunks, embeddings)

                st.session_state.indexed_embedding_model = selected_embedding_model
            st.success(f"Successfully re-indexed all documents with '{selected_embedding_model}' ({chunk_preset} preset)!")
            st.rerun()

    # Evaluation Runner Button (Phase 12)
    if st.button("🧪 Run Evaluation Benchmark", use_container_width=True, help="Runs all benchmark questions in eval/eval_questions.json and saves logs to eval/logs/."):
        indexed = get_indexed_documents()
        if not indexed:
            st.warning("Please upload documents before running evaluation.")
        else:
            with st.spinner("Running evaluation benchmark against current pipeline..."):
                eval_records = run_evaluation(
                    top_k=4,
                    use_hybrid=use_hybrid,
                    use_rerank=use_rerank,
                    chunk_preset=chunk_preset,
                    embedding_model=selected_embedding_model
                )
                if eval_records:
                    st.session_state.qa_log.extend(eval_records)
                    st.success(f"Completed evaluation benchmark! Saved {len(eval_records)} logs to eval/logs/.")
                    st.rerun()

st.sidebar.divider()

# Sidebar - Document Ingestion & Persistent Vector Store
st.sidebar.header("📁 Document Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        current_indexed = get_indexed_documents()

        if filename not in st.session_state.raw_documents:
            pages_data = extract_text_from_file(uploaded_file)
            st.session_state.raw_documents[filename] = pages_data
        else:
            pages_data = st.session_state.raw_documents[filename]

        if filename not in current_indexed:
            with st.spinner(f"Ingesting & embedding {filename}..."):
                chunks = chunk_document(pages_data, filename=filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if chunks:
                    texts_to_embed = [c["text"] for c in chunks]
                    embeddings = embed_texts(texts_to_embed, model_name=selected_embedding_model)
                    add_chunks_to_vectorstore(chunks, embeddings)
                    st.session_state.indexed_embedding_model = selected_embedding_model
            st.sidebar.success(f"Indexed {filename} ({len(chunks)} chunks)")

st.sidebar.divider()

# Sidebar - Persistent Indexed Documents View with Delete
st.sidebar.header("📚 Indexed Documents")
indexed_docs = get_indexed_documents()

if not indexed_docs:
    st.sidebar.info("No documents indexed yet.")
else:
    for doc_name, chunk_count in list(indexed_docs.items()):
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.markdown(f"📄 **{doc_name}** (`{chunk_count}` chunks)")
        if col2.button("🗑️", key=f"delete_btn_{doc_name}", help=f"Delete {doc_name} from vector store"):
            delete_document_from_vectorstore(doc_name)
            if doc_name in st.session_state.raw_documents:
                del st.session_state.raw_documents[doc_name]
            st.toast(f"Deleted {doc_name} from vector store.")
            st.rerun()

st.sidebar.divider()

# Sidebar - Q&A Logging & Export
st.sidebar.header("📊 Q&A Session Log")
log_count = len(st.session_state.qa_log)
st.sidebar.caption(f"Logged Interactions: **{log_count}**")

if log_count > 0:
    json_data = export_logs_json(st.session_state.qa_log)
    csv_data = export_logs_csv(st.session_state.qa_log)

    st.sidebar.download_button(
        label="⬇️ Download Q&A Log (JSON)",
        data=json_data,
        file_name="notebrain_qa_log.json",
        mime="application/json",
        use_container_width=True
    )
    st.sidebar.download_button(
        label="📊 Download Q&A Log (CSV)",
        data=csv_data,
        file_name="notebrain_qa_log.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.sidebar.info("Ask questions to populate the session log.")


def render_source_citations(retrieved_chunks: list):
    """
    Renders expandable '📄 Sources' section under chat answers per PRD Sections 8.2-8.3.
    """
    if not retrieved_chunks:
        return

    with st.expander("📄 Sources", expanded=False):
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "Unknown")
            page = chunk.get("page")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()

            page_str = f"page {page}" if page is not None else "page N/A"
            citation_header = f"**Source {idx}:** `{source}, {page_str}` (Similarity Score: `{score}`)"

            st.markdown(citation_header)
            st.caption(text)
            if idx < len(retrieved_chunks):
                st.divider()


# Main Area - Chat Interface
if not indexed_docs:
    st.warning("⚠️ No documents indexed yet. Please upload PDF or TXT documents in the sidebar first to start asking questions.")
else:
    # Dimension Mismatch Safety Warning Check
    model_mismatch = (selected_embedding_model != st.session_state.indexed_embedding_model)

    if model_mismatch:
        st.warning(
            f"⚠️ **Embedding Model Mismatch**: The vector store contains embeddings generated using **'{st.session_state.indexed_embedding_model}'**, "
            f"but **'{selected_embedding_model}'** is selected in Advanced Settings.\n\n"
            f"Please click **'🔄 Re-index all documents'** in the sidebar to re-embed your files before asking new questions."
        )

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "retrieved_chunks" in msg:
                render_source_citations(msg["retrieved_chunks"])

    # Chat Input
    if model_mismatch:
        st.info("💡 Re-indexing is required to synchronize vector dimensions.")
    else:
        user_query = st.chat_input("Ask a question about your documents...")

        if user_query:
            # Render User Message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Retrieve & Generate Answer
            with st.chat_message("assistant"):
                with st.spinner("Searching documents & generating grounded answer..."):
                    retrieved_chunks = retrieve_chunks(
                        user_query,
                        top_k=4,
                        use_hybrid=use_hybrid,
                        use_rerank=use_rerank,
                        embedding_model=selected_embedding_model
                    )
                    result = generate_answer(user_query, retrieved_chunks, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD)
                    answer_text = result["answer"]

                    st.markdown(answer_text)
                    render_source_citations(retrieved_chunks)

                    # Store message in session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "retrieved_chunks": retrieved_chunks,
                        "is_fallback": result["is_fallback"],
                        "top_score": result["top_score"]
                    })

                    # Append Q&A Log Record
                    log_rec = create_log_record(
                        question=user_query,
                        answer=answer_text,
                        is_fallback=result["is_fallback"],
                        top_score=result["top_score"],
                        retrieved_chunks=retrieved_chunks,
                        settings={
                            "top_k": 4,
                            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                            "hybrid_search": use_hybrid,
                            "reranking": use_rerank,
                            "chunk_size_preset": chunk_preset,
                            "embedding_model": selected_embedding_model
                        }
                    )
                    st.session_state.qa_log.append(log_rec)
                    st.rerun()
