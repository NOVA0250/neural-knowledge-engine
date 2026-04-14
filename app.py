import streamlit as st
import tempfile, os, time
import numpy as np
import pandas as pd
import pypdf

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIG ─────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

# ── LLM SETUP ─────────────────────────────────────────
PROVIDER_MODELS = {
    "Groq": "llama-3.3-70b-versatile",
    "Gemini": "gemini-2.0-flash",
    "OpenAI": "gpt-4o-mini",
}

def get_llm(provider, api_key):
    if provider == "Groq":
        from groq import Groq
        return {"provider": "groq", "client": Groq(api_key=api_key)}
    elif provider == "Gemini":
        from google import genai
        return {"provider": "gemini", "client": genai.Client(api_key=api_key)}
    else:
        from openai import OpenAI
        return {"provider": "openai", "client": OpenAI(api_key=api_key)}

def _is_rate_limit(e):
    return any(x in str(e).lower() for x in ["429", "quota", "resource_exhausted"])

def llm_complete(llm, prompt):
    try:
        if llm["provider"] == "gemini":
            return llm["client"].models.generate_content(
                model=PROVIDER_MODELS["Gemini"], contents=prompt
            ).text

        resp = llm["client"].chat.completions.create(
            model=PROVIDER_MODELS["Groq"],
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    except Exception as e:
        # 🔥 AUTO FALLBACK TO GROQ
        if _is_rate_limit(e):
            from groq import Groq
            groq = Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))
            resp = groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        raise e

# ── PDF PROCESSING ────────────────────────────────────
def extract_chunks(file):
    reader = pypdf.PdfReader(file)
    chunks = []

    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            chunk = text[start:start + CHUNK_SIZE]
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

# ── EMBEDDINGS (LOCAL FREE) ───────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks):
    model = load_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    return embeddings

def retrieve(query, chunks, embeddings):
    model = load_model()
    q_emb = model.encode([query])

    scores = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:TOP_K]

    return [chunks[i] for i in top_idx]

# ── ANSWER ────────────────────────────────────────────
def answer(query, chunks, embeddings, llm):
    docs = retrieve(query, chunks, embeddings)

    context = "\n\n".join(docs)

    prompt = f"""
Answer using ONLY this context:

{context}

Question: {query}
"""

    try:
        return llm_complete(llm, prompt)
    except:
        return "\n\n".join(docs[:3])  # fallback

# ── STREAMLIT UI ──────────────────────────────────────
st.title("⚡ Neural Knowledge Engine (FINAL)")

provider = st.selectbox("Provider", ["Groq", "Gemini", "OpenAI"])
api_key = st.text_input("API Key", type="password")

uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Initialize"):
    if not uploaded:
        st.error("Upload files")
    else:
        chunks = []
        for f in uploaded:
            chunks += extract_chunks(f)

        embeddings = build_index(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.llm = get_llm(provider, api_key)

        st.success("Engine Ready")

# ── CHAT ─────────────────────────────────────────────
if "chunks" in st.session_state:
    query = st.chat_input("Ask anything")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            ans = answer(
                query,
                st.session_state.chunks,
                st.session_state.embeddings,
                st.session_state.llm
            )
            st.write(ans)
