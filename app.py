import streamlit as st
import os
import tempfile
import numpy as np
import faiss
import pypdf

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

# ── CONFIG ─────────────────────────────────────────────────────
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

# ── INIT ───────────────────────────────────────────────────────
st.set_page_config(page_title="Neural Knowledge Engine", layout="wide")

# Load models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embed_model

embed_model = load_models()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ── UI STYLE ───────────────────────────────────────────────────
st.markdown("""
<style>
body {background:#050507;color:#d1d5db;}
.stButton > button:hover {box-shadow:0 0 12px cyan;}
[data-testid="stChatMessage"]:hover {box-shadow:0 0 12px cyan;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>⚡ Neural Knowledge Engine</h1>", unsafe_allow_html=True)

# ── CHUNKING ───────────────────────────────────────────────────
def extract_chunks(file_bytes, filename):
    chunks, meta = [], []
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file_bytes)
        tmp = f.name

    reader = pypdf.PdfReader(tmp)
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        text = " ".join(text.split())

        start = 0
        while start < len(text):
            piece = text[start:start+CHUNK_SIZE]
            if len(piece) > 60:
                chunks.append(piece)
                meta.append({"source": filename, "page": i+1})
            start += CHUNK_SIZE - CHUNK_OVERLAP

    os.remove(tmp)
    return chunks, meta

# ── BUILD INDEX ─────────────────────────────────────────────────
def build_index(files):
    all_chunks, all_meta = [], []

    for f in files:
        c, m = extract_chunks(f.getvalue(), f.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if len(all_chunks) > 150:
        all_chunks = all_chunks[:150]
        all_meta = all_meta[:150]

    embeddings = embed_model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # keyword index
    vectorizer = TfidfVectorizer().fit(all_chunks)
    tfidf_matrix = vectorizer.transform(all_chunks)

    return index, all_chunks, all_meta, vectorizer, tfidf_matrix

# ── HYBRID RETRIEVE ─────────────────────────────────────────────
def retrieve(query, index, chunks, meta, vectorizer, tfidf_matrix):
    # semantic
    q_vec = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)
    _, sem_ids = index.search(q_vec, 20)

    # keyword
    q_tfidf = vectorizer.transform([query])
    scores = (tfidf_matrix @ q_tfidf.T).toarray().ravel()
    kw_ids = np.argsort(scores)[::-1][:20]

    combined = list(set(sem_ids[0]) | set(kw_ids))

    docs = [{"text": chunks[i], **meta[i]} for i in combined if i != -1]

    return docs[:TOP_K]

# ── CACHE LLM ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_llm(prompt):
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ── ANSWER ─────────────────────────────────────────────────────
def generate_answer(query, docs):
    context = "\n\n".join(d["text"] for d in docs)

    prompt = f"""
Answer ONLY from context:

{context}

Question: {query}
"""

    try:
        return cached_llm(prompt)
    except:
        # fallback
        if docs:
            return "⚠️ API busy. Local answer:\n\n" + docs[0]["text"][:500]
        return "⚠️ No data found."

# ── SIDEBAR ────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Initialize"):
    idx, chunks, meta, vec, tfidf = build_index(uploaded)

    st.session_state.index = idx
    st.session_state.chunks = chunks
    st.session_state.meta = meta
    st.session_state.vectorizer = vec
    st.session_state.tfidf = tfidf

    st.sidebar.success("✅ Ready")

# ── CHAT ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    docs = retrieve(
        prompt,
        st.session_state.index,
        st.session_state.chunks,
        st.session_state.meta,
        st.session_state.vectorizer,
        st.session_state.tfidf
    )

    with st.chat_message("assistant"):
        with st.spinner("⚡ Thinking..."):
            ans = generate_answer(prompt, docs)
            st.markdown(ans)

            if docs:
                sources = set([f"{d['source']} p.{d['page']}" for d in docs])
                st.markdown("<br>".join([f"📄 {s}" for s in sources]), unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": ans})
