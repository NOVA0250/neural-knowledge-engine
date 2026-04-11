import streamlit as st
import os
import tempfile

# --- MODERN LANGCHAIN IMPORTS (BATTLE-TESTED) ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- UI CONFIGURATION ---
st.set_page_config(page_title="NEURAL KNOWLEDGE ENGINE", layout="wide")

def apply_neon_theme():
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

apply_neon_theme()

# --- BACKEND LOGIC ---
def process_documents(uploaded_files, openai_api_key):
    all_docs = []
    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = text_splitter.split_documents(docs)
                all_docs.extend(splits)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Engine Error: {str(e)}")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ ENGINE CORE")

    # Prefer secret from Streamlit Cloud, fall back to user input
    default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("OpenAI API Key", value=default_key, type="password")

    st.divider()
    uploaded_files = st.file_uploader("Ingest Research Papers (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("INITIALIZE ENGINE"):
        if not api_key:
            st.error("Enter API Key")
        elif not uploaded_files:
            st.error("Upload PDFs")
        else:
            with st.spinner("Decoding Data..."):
                vs = process_documents(uploaded_files, api_key)
                if vs:
                    st.session_state.vectorstore = vs
                    st.success("Sync Complete.")

# --- MAIN INTERFACE ---
st.title("⚡ AI KNOWLEDGE ENGINE")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query..."):
    if not api_key:
        st.error("Missing API Key.")
    elif "vectorstore" not in st.session_state:
        st.error("Initialize documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing..."):
                try:
                    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "Answer based only on context: {context}"),
                        ("human", "{input}"),
                    ])
                    
                    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(
                        st.session_state.vectorstore.as_retriever(), combine_docs_chain
                    )
                    
                    response = rag_chain.invoke({"input": prompt})
                    
                    answer = response["answer"]
                    sources = list(set([
                        doc.metadata.get("source", "Unknown") for doc in response["context"]
                    ]))
                    
                    full_res = f"{answer}\n\n**SOURCES:**\n" + "\n".join([f"- {s}" for s in sources])
                    st.markdown(full_res)
                    st.session_state.messages.append({"role": "assistant", "content": full_res})
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
