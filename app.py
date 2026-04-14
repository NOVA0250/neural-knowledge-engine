import streamlit as st
import os
import tempfile
import numpy as np
import faiss
import pypdf

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional OpenAI
try:
    from openai import OpenAI
    USE_OPENAI = True
except:
    USE_OPENAI = False


# ── CONFIG ─────────────────────────────────────────────────────
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 10   # get more → rerank later

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100


# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(page_title="Neural Knowledge Engine", layout="wide")

# ── UI ─────────────────────────────────────────────────────────
st.markdown("""
<style>
body {background:#050507;color:#d1d5db;}
.stButton > button:hover {box-shadow:0 0 12px cyan;}
[data-testid="stChatMessage"]:hover {box-shadow:0 0 12px cyan;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>⚡ Neural Knowledge Engine</h1>", unsafe_allow_html=True)


# ── LOAD MODELS ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embed_model = load_models()

if USE_OPENAI:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except:
        USE_OPENAI = False


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


# ── BUILD INDEX ────────────────────────────────────────────────
def build_index(files):
    all_chunks, all_meta = [], []

    for f in files:
        c, m = extract_chunks(f.getvalue(), f.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if len(all_chunks) > 200:
        all_chunks = all_chunks[:200]
        all_meta = all_meta[:200]

    embeddings = embed_model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    vectorizer = TfidfVectorizer().fit(all_chunks)
    tfidf_matrix = vectorizer.transform(all_chunks)

    return index, all_chunks, all_meta, vectorizer, tfidf_matrix


# ── RETRIEVE ───────────────────────────────────────────────────
def retrieve(query, index, chunks, meta, vectorizer, tfidf_matrix):
    q_vec = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)

    _, sem_ids = index.search(q_vec, 20)

    q_tfidf = vectorizer.transform([query])
    scores = (tfidf_matrix @ q_tfidf.T).toarray().ravel()
    kw_ids = np.argsort(scores)[::-1][:20]

    combined = list(set(sem_ids[0]) | set(kw_ids))

    docs = [{"text": chunks[i], **meta[i]} for i in combined if i != -1]

    return docs


# ── RERANK ─────────────────────────────────────────────────────
def rerank(query, docs):
    q_vec = embed_model.encode([query])[0]

    scored = []
    for d in docs:
        d_vec = embed_model.encode([d["text"]])[0]
        score = np.dot(q_vec, d_vec)
        scored.append((score, d))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored[:5]]


# ── COMPRESS CONTEXT ───────────────────────────────────────────
def compress_context(query, docs):
    keywords = query.lower().split()

    refined = []
    for d in docs:
        sentences = d["text"].split(".")
        best = [s for s in sentences if any(k in s.lower() for k in keywords)]
        refined.append(" ".join(best[:2]))

    return refined


# ── LOCAL ANSWER ───────────────────────────────────────────────
def local_answer(query, docs):
    text = "\n\n".join(d["text"] for d in docs[:2])

    return f"""
📌 Answer (Local Intelligence Mode):

{text[:600]}

🔎 Based on retrieved context.
"""


# ── OPENAI ANSWER ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def openai_answer(prompt):
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# ── GENERATE ANSWER ────────────────────────────────────────────
def generate_answer(query, docs):
    docs = rerank(query, docs)
    compressed = compress_context(query, docs)

    context = "\n\n".join(compressed)

    prompt = f"""
Answer clearly using context:

{context}

Question: {query}
"""

    if USE_OPENAI:
        try:
            return openai_answer(prompt)
        except:
            return local_answer(query, docs)

    return local_answer(query, docs)


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
                sources = set([f"{d['source']} p.{d['page']}" for d in docs[:5]])
                st.markdown("<br>".join([f"📄 {s}" for s in sources]), unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": ans})
