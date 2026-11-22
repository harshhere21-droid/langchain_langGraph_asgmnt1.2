"""Streamlit interface for the LlamaIndex + LangChain document Q&A system."""

from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from openai import OpenAIError
from pydantic import ValidationError
from pypdf.errors import PdfReadError

from document_qa import answer_question, build_index, extract_documents, retrieve_context

load_dotenv()
st.set_page_config(page_title="DocuMind | Document Q&A", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    .hero { padding: 1.35rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#132238,#245d72); color: white; margin-bottom: 1rem; }
    .hero h1 { margin: 0 0 .25rem 0; color: white; }
    .hero p { margin: 0; color: #d7e8ee; }
    [data-testid="stMetric"] { background: white; padding: 16px; border-radius: 12px; border: 1px solid #e6eaf0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>DocuMind</h1><p>Ask questions across your files and public web pages, with answers grounded in retrieved passages.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1 · Add sources")
    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="PDF, plain text, or Markdown; up to 10 MB each.",
    )
    url_text = st.text_area(
        "Public web-page URLs (one per line)",
        placeholder="https://example.org/research",
        help="Only public HTTP(S) web pages returning HTML or plain text are accepted.",
    )
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=10, value=5)
    index_button = st.button("Index sources", type="primary", use_container_width=True)
    st.caption("Local Hugging Face embeddings; first run downloads the embedding model.")

st.info(
    "Documents and vector indexes are held in this session only. Retrieved passages are sent to "
    "OpenAI for answer generation. Set `OPENAI_API_KEY` in `.env`."
)

if index_button:
    file_payloads = [(file.name, file.getvalue()) for file in (uploaded or [])]
    urls = [line.strip() for line in url_text.splitlines() if line.strip()]
    try:
        with st.spinner("Reading sources and creating the semantic index…"):
            documents = extract_documents(file_payloads, urls)
            index = build_index(documents)
        st.session_state["docqa_index"] = index
        st.session_state["docqa_sources"] = sorted({str(doc.metadata.get("source", "Unknown")) for doc in documents})
        st.session_state.pop("docqa_last_result", None)
        st.success(f"Indexed {len(documents)} source section(s) from {len(st.session_state['docqa_sources'])} source(s).")
    except (ValueError, RuntimeError, OSError, requests.RequestException, PdfReadError) as exc:
        st.error(f"Could not index sources: {exc}")

index = st.session_state.get("docqa_index")
if index is None:
    st.markdown(
        """
        ### How to get started
        1. Upload one or more PDFs, TXT, or Markdown files, or enter public web URLs.
        2. Select **Index sources**. The app extracts text and builds a LlamaIndex vector index.
        3. Ask a question or summarize the indexed collection. Answers include source and page references.
        """
    )
    st.stop()

st.subheader("Indexed sources")
st.write(" · ".join(st.session_state.get("docqa_sources", [])))
question = st.text_area(
    "Ask a question",
    placeholder="What are the main findings across these documents?",
    height=90,
)
col_ask, col_summary, _ = st.columns([1, 1, 4])
ask_clicked = col_ask.button("Answer question", type="primary")
summary_clicked = col_summary.button("Summarize sources")

if ask_clicked or summary_clicked:
    submitted_question = (
        "Summarize the main topics and key points across all indexed sources."
        if summary_clicked
        else question.strip()
    )
    if not submitted_question:
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer…"):
                passages = retrieve_context(index, submitted_question, top_k=top_k)
                response = answer_question(submitted_question, passages)
            st.session_state["docqa_last_result"] = {
                "question": submitted_question,
                "response": response,
                "passages": passages,
            }
        except (ValueError, RuntimeError, OpenAIError, OutputParserException, ValidationError) as exc:
            st.error(f"Could not answer the question: {exc}")

result = st.session_state.get("docqa_last_result")
if result:
    response = result["response"]
    st.divider()
    st.subheader("Answer")
    if response.insufficient_evidence:
        st.warning(response.answer)
    else:
        st.markdown(response.answer)
    if response.citations:
        citation_passages = [p for p in result["passages"] if p.source_id in response.citations]
        st.markdown("**Sources**")
        for passage in citation_passages:
            st.markdown(f"- **[{passage.source_id}]** {passage.source} · {passage.page}")
    with st.expander("Retrieved passages"):
        for passage in result["passages"]:
            score_text = f" · similarity {passage.score:.3f}" if passage.score is not None else ""
            st.markdown(f"**[{passage.source_id}] {passage.source} · {passage.page}{score_text}**")
            st.write(passage.text)
            st.divider()

st.caption(
    "Answers can be incomplete or incorrect. Verify important claims against the cited source documents. "
    "Retrieved document text is treated as untrusted input."
)
