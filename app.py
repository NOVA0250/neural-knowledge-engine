import streamlit as st
import os
import tempfile
from pdf_utils import load_and_chunk_pdfs
from embeddings import EmbeddingManager
from retrieval import HybridRetriever
from qa import QASystem

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="NEURAL KNOWLEDGE ENGINE",
    page_icon="🧠",
    layout="wide"
)

# -------------------- CUSTOM CSS (SPIDERMAN UI) --------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
    background: radial-gradient(circle at top, #0a0f2c, #000000);
    color: white;
}

/* Title Glow */
.title {
    font-size: 42px;
    font-weight: 600;
    text-align: center;
    color: #ff1e1e;
    text-shadow: 0 0 15px #ff0000, 0 0 30px #1e90ff;
    margin-bottom: 10px;
}

/* Cards */
.card {
    background: rgba(10, 15, 50, 0.7);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,0,0,0.3);
    transition: 0.3s;
}

.card:hover {
    box-shadow: 0 0 20px #ff0000, 0 0 40px #1e90ff;
    transform: scale(1.02);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    border-radius: 12px;
    color: white;
    border: 1px solid #ff0000;
    transition: 0.3s;
}

.stButton>button:hover {
    box-shadow: 0 0 15px red, 0 0 25px blue;
    transform: scale(1.05);
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
    background: rgba(20,20,50,0.6);
    border: 1px solid rgba(255,0,0,0.3);
    transition: 0.3s;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: 0 0 10px red, 0 0 20px blue;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #050816;
    border-right: 1px solid rgba(255,0,0,0.2);
}

/* HUD divider */
hr {
    border: 1px solid rgba(255,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">🧠 NEURAL KNOWLEDGE ENGINE</div>', unsafe_allow_html=True)
st.markdown("### ⚡ Hybrid Intelligence • FAISS + BM25 • Real-Time Reasoning")

# -------------------- SESSION STATE --------------------
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# -------------------- API KEY --------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("Add GROQ_API_KEY in Streamlit secrets")
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("## 🕷 Upload Intelligence Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# -------------------- PROCESS FILES --------------------
if uploaded_files:
    current_files = [f.name for f in uploaded_files]

    if current_files != st.session_state.processed_files:
        with st.spinner("⚡ Neural Processing..."):
            temp_paths = []

            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    temp_paths.append(tmp.name)

            documents = load_and_chunk_pdfs(temp_paths)

            for p in temp_paths:
                os.unlink(p)

            st.session_state.documents = documents

            embedding_manager = EmbeddingManager()
            embedding_manager.build_index(documents)

            retriever = HybridRetriever(embedding_manager, documents)
            qa_system = QASystem(groq_api_key, retriever)

            st.session_state.embedding_manager = embedding_manager
            st.session_state.retriever = retriever
            st.session_state.qa_system = qa_system
            st.session_state.processed_files = current_files
            st.session_state.chat_history = []

# -------------------- MAIN UI --------------------
if st.session_state.documents:

    st.sidebar.markdown("### 📊 System Stats")
    st.sidebar.write(f"Chunks: {len(st.session_state.documents)}")
    st.sidebar.write(f"Files: {len(st.session_state.processed_files)}")

    st.markdown("## 💬 Neural Chat Interface")

    # Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    question = st.chat_input("Ask anything...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""

            for chunk in st.session_state.qa_system.answer_question(question):
                response += chunk
                placeholder.markdown(response + "▌")

            placeholder.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.sidebar.button("🧹 Reset System"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.markdown("""
    <div class="card">
        <h3>🚀 System Capabilities</h3>
        <ul>
            <li>Hybrid Retrieval (FAISS + BM25)</li>
            <li>Context-aware AI reasoning</li>
            <li>Streaming responses</li>
            <li>Multi-PDF Intelligence</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.sidebar.markdown("---")
st.sidebar.markdown("⚡ Powered by Groq + FAISS + Transformers")
