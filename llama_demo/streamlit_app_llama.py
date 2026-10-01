import io
import re
import threading

import streamlit as st
from llama_cpp import Llama
from pypdf import PdfReader
from rank_bm25 import BM25Plus

CHUNK_WORDS = 150    # words per chunk of PDF text
OVERLAP_WORDS = 30   # overlap so sentences at a chunk edge aren't lost
TOP_K = 3            # chunks passed to the model per question

### Load a small Llama that runs inside the app itself: no API key, no secrets.
# Llama 3.2 1B, compressed to 4 bits per weight (~0.8 GB), so it fits in Community Cloud's memory limit.
# n_ctx raised from the 512-token default so the retrieved PDF text fits in the prompt.
@st.cache_resource(show_spinner="Downloading Llama 3.2 1B (about 800 MB). Only the first visitor waits for this...")
def load_model():
    llm = Llama.from_pretrained(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="*Q4_K_M.gguf",
        n_threads=2,
        n_ctx=2048,
        verbose=False,
    )
    return llm, threading.Lock()  # one model shared by every visitor, answering one request at a time

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "which", "who",
    "with", "how", "why", "when", "does", "do", "did", "can", "about",
}

def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]

### Split the uploaded PDFs into overlapping chunks and build a keyword (BM25) index.
# Cached on the file contents, so the PDFs are only parsed again when the upload changes.
@st.cache_data(show_spinner="Reading PDF...")
def build_index(files):
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for name, data in files:
        for page_no, page in enumerate(PdfReader(io.BytesIO(data)).pages, start=1):
            words = (page.extract_text() or "").split()
            for start in range(0, len(words), step):
                chunks.append({"source": name, "page": page_no,
                               "text": " ".join(words[start:start + CHUNK_WORDS])})
                if start + CHUNK_WORDS >= len(words):
                    break
    if not chunks:
        return [], None
    return chunks, BM25Plus([tokenize(c["text"]) for c in chunks])

def retrieve(query, chunks, bm25, k=TOP_K):
    scores = bm25.get_scores(tokenize(query))
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in best]

llm, lock = load_model()

st.title("Ask your PDF")
uploaded = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
query = st.text_input("Your question")

chunks, bm25 = [], None
if uploaded:
    chunks, bm25 = build_index(tuple((f.name, f.getvalue()) for f in uploaded))
    if not chunks:
        st.warning("No text found in the PDF. Scanned/image-only PDFs need OCR first.")

if st.button("Ask", disabled=not query.strip()):
    hits = retrieve(query, chunks, bm25) if bm25 else []
    if hits:
        context = "\n\n".join(f"[{i}] ({h['source']}, p.{h['page']}) {h['text']}" for i, h in enumerate(hits, 1))
        system = ("Answer the question using only the context below. "
                  "If the answer is not in the context, say you could not find it in the PDF. "
                  "Keep the answer short.\n\nContext:\n" + context)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    else:
        messages = [{"role": "user", "content": query}]  # no PDF: plain chat

    with st.spinner("Llama is thinking on this app's CPU..."), lock:
        reply = llm.create_chat_completion(messages=messages, max_tokens=256)
    st.write(reply["choices"][0]["message"]["content"])

    if hits:
        with st.expander("Retrieved passages"):
            for h in hits:
                st.markdown(f"**{h['source']}, page {h['page']}**")
                st.caption(h["text"])
