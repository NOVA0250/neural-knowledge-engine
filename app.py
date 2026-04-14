import streamlit as st
import os
import tempfile
import time

from google import genai
from google.genai import types
import faiss
import numpy as np
import pypdf

# ✅ LOCAL EMBEDDINGS
from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHAT_MODEL  = "gemini-1.5-flash"   # ✅ cheaper + stable

CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 100
EMBED_BATCH   = 100
TOP_K         = 5

# Load local embedding model once
local_model = SentenceTransformer('all-MiniLM-L6-v2')


# ── CLIENT ────────────────────────────────────────────────────────────────────
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


# ── PDF CHUNKING ──────────────────────────────────────────────────────────────
def extract_chunks(file_bytes: bytes, filename: str):
    chunks, meta = [], []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        tmp = f.name

    try:
        reader = pypdf.PdfReader(tmp)
        for page_num, page in enumerate(reader.pages):
            raw = (page.extract_text() or "").strip()
            if not raw:
                continue

            text = " ".join(raw.split())

            start = 0
            while start < len(text):
                piece = text[start:start + CHUNK_SIZE].strip()

                if len(piece) > 60:
                    chunks.append(piece)
                    meta.append({
                        "source": filename,
                        "page": page_num + 1
                    })

                start += CHUNK_SIZE - CHUNK_OVERLAP
    finally:
        os.remove(tmp)

    return chunks, meta


# ── LOCAL EMBEDDING (ZERO API) ─────────────────────────────────────────────────
def embed_batch(texts: list) -> np.ndarray:
    embeddings = local_model.encode(texts)
    return np.array(embeddings, dtype="float32")


# ── BUILD INDEX ───────────────────────────────────────────────────────────────
def build_index(uploaded_files):
    all_chunks, all_meta = [], []

    for uf in uploaded_files:
        c, m = extract_chunks(uf.getvalue(), uf.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if not all_chunks:
        return None, None, None, "No readable text found."

    # 🔥 LIMIT chunks (prevents overload)
    if len(all_chunks) > 150:
        all_chunks = all_chunks[:150]
        all_meta = all_meta[:150]

    progress = st.progress(0, text="Embedding locally...")

    all_vecs = []

    for i in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[i:i + EMBED_BATCH]
        vecs = embed_batch(batch)
        all_vecs.append(vecs)

        pct = min((i + EMBED_BATCH) / len(all_chunks), 1.0)
        progress.progress(pct)

    progress.empty()

    matrix = np.vstack(all_vecs)
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    return index, all_chunks, all_meta, None


# ── RETRIEVE ──────────────────────────────────────────────────────────────────
def retrieve(query, index, chunks, meta):
    q_vec = embed_batch([query])
    faiss.normalize_L2(q_vec)

    _, ids = index.search(q_vec, TOP_K)

    return [
        {"text": chunks[i], **meta[i]}
        for i in ids[0] if i != -1
    ]


# ── CACHED GEMINI RESPONSE ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_generate_answer(prompt: str, api_key: str):
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    return response.text


# ── GENERATE ANSWER ───────────────────────────────────────────────────────────
def generate_answer(query, docs, api_key):
    context = "\n\n---\n\n".join(
        f"[{d['source']} | Page {d['page']}]\n{d['text']}"
        for d in docs
    )

    prompt = f"""
You are an expert assistant.

Answer ONLY from the context below.
Be concise and clear.

CONTEXT:
{context}

QUESTION:
{query}
"""

    try:
        return cached_generate_answer(prompt, api_key)
    except Exception:
        return "⚠️ API busy or quota issue. Try again later."


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neural Knowledge Engine", layout="wide")

st.title("⚡ Neural Knowledge Engine")

api_key = st.sidebar.text_input("Gemini API Key", type="password")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

if st.sidebar.button("Initialize"):
    if not uploaded_files:
        st.error("Upload PDFs first.")
    else:
        idx, chunks, meta, err = build_index(uploaded_files)

        if err:
            st.error(err)
        else:
            st.session_state.index = idx
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.api_key = api_key
            st.success("Index built successfully 🚀")


# ── CHAT ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):
    if "index" not in st.session_state:
        st.error("Initialize first.")
    elif not api_key:
        st.error("Enter API key.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            docs = retrieve(
                prompt,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.meta
            )

            answer = generate_answer(
                prompt,
                docs,
                st.session_state.api_key
            )

            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )


# ── CLEAR CHAT ────────────────────────────────────────────────────────────────
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
