import streamlit as st
import os
import tempfile

import openai
import faiss
import numpy as np
import pypdf

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NEURAL KNOWLEDGE ENGINE", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0a0a0c; color: #e0e0e0; }
h1, h2, h3 {
    color: #00f2ff !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 10px #00f2ff55;
}
[data-testid="stSidebar"] {
    background-color: #111114;
    border-right: 1px solid #7000ff;
}
.stTextInput > div > div > input {
    background-color: #1a1a1e;
    color: #00f2ff;
    border: 1px solid #7000ff;
}
.stButton > button {
    background: linear-gradient(45deg, #7000ff, #00f2ff);
    color: white; border: none; border-radius: 5px;
    font-weight: bold; width: 100%; transition: 0.3s;
}
.stButton > button:hover {
    box-shadow: 0 0 20px #7000ffaa;
    transform: scale(1.02);
}
.stChatMessage {
    background-color: #16161a;
    border: 1px solid #333;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def extract_pages(file_bytes, filename):
    chunks, meta = [], []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        reader = pypdf.PdfReader(tmp)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            # simple fixed-size chunking with overlap
            size, overlap = 800, 150
            start = 0
            while start < len(text):
                piece = text[start:start + size]
                if piece.strip():
                    chunks.append(piece)
                    meta.append({"source": filename, "page": i + 1})
                start += size - overlap
    finally:
        os.remove(tmp)
    return chunks, meta


def embed(texts, client):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")


def build_index(uploaded_files, client):
    all_chunks, all_meta = [], []
    for uf in uploaded_files:
        c, m = extract_pages(uf.getvalue(), uf.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if not all_chunks:
        st.error("No text could be extracted from the PDFs.")
        return None, None, None

    vecs = []
    batch = 500
    for i in range(0, len(all_chunks), batch):
        vecs.append(embed(all_chunks[i:i+batch], client))
    matrix = np.vstack(vecs)
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, all_chunks, all_meta


def retrieve(query, index, chunks, meta, client, k=5):
    q = embed([query], client)
    faiss.normalize_L2(q)
    _, ids = index.search(q, k)
    return [{"text": chunks[i], **meta[i]} for i in ids[0] if i != -1]


def answer(query, docs, client):
    ctx = "\n\n---\n\n".join(
        f"[{d['source']} | p.{d['page']}]\n{d['text']}" for d in docs
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content":
                "You are a precise research assistant. "
                "Answer using ONLY the context below. "
                "If the answer isn't there, say so.\n\nCONTEXT:\n" + ctx},
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ ENGINE CORE")

    default_key = ""
    try:
        default_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass

    api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
    st.divider()
    uploaded_files = st.file_uploader(
        "Ingest Research Papers (PDF)", type="pdf", accept_multiple_files=True
    )

    if st.button("INITIALIZE ENGINE"):
        if not api_key:
            st.error("Enter API Key.")
        elif not uploaded_files:
            st.error("Upload at least one PDF.")
        else:
            client = openai.OpenAI(api_key=api_key)
            with st.spinner("Parsing & indexing…"):
                idx, chunks, meta = build_index(uploaded_files, client)
            if idx is not None:
                st.session_state.update(
                    index=idx, chunks=chunks, meta=meta, api_key=api_key
                )
                st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.title("⚡ AI KNOWLEDGE ENGINE")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Query the documents…"):
    if not api_key:
        st.error("Enter your OpenAI API Key in the sidebar.")
    elif "index" not in st.session_state:
        st.error("Upload PDFs and click INITIALIZE ENGINE first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing…"):
                try:
                    client = openai.OpenAI(api_key=st.session_state.api_key)
                    docs = retrieve(prompt, st.session_state.index,
                                    st.session_state.chunks, st.session_state.meta, client)
                    ans = answer(prompt, docs, client)
                    srcs = list({f"{d['source']} (p.{d['page']})" for d in docs})
                    full = ans + "\n\n**SOURCES:**\n" + "\n".join(f"- {s}" for s in srcs)
                    st.markdown(full)
                    st.session_state.messages.append({"role": "assistant", "content": full})
                except openai.AuthenticationError:
                    st.error("Invalid API key — check and try again.")
                except Exception as e:
                    st.error(f"Error: {e}")
