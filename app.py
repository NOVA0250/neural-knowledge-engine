import streamlit as st
import os
import tempfile

from google import genai
import faiss
import numpy as np
import pypdf
from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHAT_MODEL = "gemini-1.5-flash"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
EMBED_BATCH = 100
TOP_K = 5

# Load local embedding model
local_model = SentenceTransformer('all-MiniLM-L6-v2')


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neural Knowledge Engine", layout="wide")

# ── UI STYLE (GLOW + GLASS) ───────────────────────────────────────────────────
st.markdown("""
<style>
html, body, .stApp {
    background: radial-gradient(circle at top, #0a0a12, #050507);
    color: #d1d5db;
}

:root {
    --glow: 0 0 12px rgba(0,255,255,0.6),
            0 0 24px rgba(0,255,255,0.3);
}

.stButton > button {
    background: linear-gradient(135deg, #6a00ff, #00f0ff);
    color: white;
    border-radius: 10px;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    box-shadow: var(--glow);
    transform: translateY(-2px);
}

[data-testid="stSidebar"] {
    background: rgba(15,15,25,0.8);
    backdrop-filter: blur(10px);
}

[data-testid="stChatMessage"] {
    background: rgba(20,20,35,0.7);
    border-radius: 14px;
    padding: 12px;
    transition: 0.25s;
}
[data-testid="stChatMessage"]:hover {
    box-shadow: var(--glow);
}

[data-testid="stChatInput"] textarea {
    background: #0a0a10 !important;
    color: #00f0ff !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea:focus {
    box-shadow: var(--glow);
}

h1 {
    background: linear-gradient(90deg, #00f0ff, #6a00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── TITLE ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>⚡ Neural Knowledge Engine</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>RAG • Semantic Search • AI Intelligence</p>",
    unsafe_allow_html=True
)

# ── CLIENT ────────────────────────────────────────────────────────────────────
def get_client(api_key):
    return genai.Client(api_key=api_key)


# ── PDF CHUNKING ──────────────────────────────────────────────────────────────
def extract_chunks(file_bytes, filename):
    chunks, meta = [], []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        tmp = f.name

    try:
        reader = pypdf.PdfReader(tmp)
        for page_num, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            text = " ".join(text.split())
            start = 0
            while start < len(text):
                piece = text[start:start+CHUNK_SIZE]
                if len(piece) > 60:
                    chunks.append(piece)
                    meta.append({"source": filename, "page": page_num+1})
                start += CHUNK_SIZE - CHUNK_OVERLAP
    finally:
        os.remove(tmp)

    return chunks, meta


# ── LOCAL EMBEDDINGS ──────────────────────────────────────────────────────────
def embed_batch(texts):
    return np.array(local_model.encode(texts), dtype="float32")


# ── BUILD INDEX ───────────────────────────────────────────────────────────────
def build_index(files):
    all_chunks, all_meta = [], []
    for f in files:
        c, m = extract_chunks(f.getvalue(), f.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if not all_chunks:
        return None, None, None

    if len(all_chunks) > 150:
        all_chunks = all_chunks[:150]
        all_meta = all_meta[:150]

    vecs = embed_batch(all_chunks)
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    return index, all_chunks, all_meta


# ── RETRIEVE ──────────────────────────────────────────────────────────────────
def retrieve(query, index, chunks, meta):
    q = embed_batch([query])
    faiss.normalize_L2(q)
    _, ids = index.search(q, TOP_K)

    return [{"text": chunks[i], **meta[i]} for i in ids[0] if i != -1]


# ── CACHE ANSWER ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_answer(prompt, api_key):
    client = genai.Client(api_key=api_key)
    res = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return res.text


# ── GENERATE ANSWER ───────────────────────────────────────────────────────────
def generate_answer(query, docs, api_key):
    context = "\n\n---\n\n".join(
        f"[{d['source']} | Page {d['page']}]\n{d['text']}" for d in docs
    )

    prompt = f"""
Answer only using context.

{context}

Question: {query}
"""

    try:
        return cached_answer(prompt, api_key)
    except:
        return "⚠️ API busy or quota issue."


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
api_key = st.sidebar.text_input("Gemini API Key", type="password")

uploaded = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if "index" in st.session_state:
    st.sidebar.success("🟢 Engine Ready")
else:
    st.sidebar.warning("🔴 Not Initialized")

if st.sidebar.button("Initialize"):
    if not uploaded:
        st.error("Upload PDFs first.")
    else:
        idx, chunks, meta = build_index(uploaded)
        st.session_state.index = idx
        st.session_state.chunks = chunks
        st.session_state.meta = meta
        st.session_state.api_key = api_key
        st.success("Index built 🚀")


# ── CHAT ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask something..."):
    if "index" not in st.session_state:
        st.error("Initialize first")
    elif not api_key:
        st.error("Enter API key")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("⚡ Thinking..."):
                docs = retrieve(prompt,
                                st.session_state.index,
                                st.session_state.chunks,
                                st.session_state.meta)

                ans = generate_answer(prompt, docs, st.session_state.api_key)
                st.markdown(ans)

                # sources
                if docs:
                    sources = set([f"{d['source']} p.{d['page']}" for d in docs])
                    st.markdown("<br>".join([f"📄 {s}" for s in sources]), unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": ans})


# ── CLEAR ─────────────────────────────────────────────────────────────────────
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
