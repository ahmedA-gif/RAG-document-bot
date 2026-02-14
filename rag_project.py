import os
import shutil
import logging
from pathlib import Path

import streamlit as st
import requests
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain # fixed import
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # <-- new import for Groq

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------
# Constants (can be overridden via .env)
# ------------------------------
DOCS_DIR = os.getenv("DOCS_DIR", "knowledge_base")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
# Groq model (default to a fast, capable one)
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Sentence-transformers model for embeddings
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
DEFAULT_K_RETRIEVAL = int(os.getenv("K_RETRIEVAL", 6))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))

# Create directories if not exist
os.makedirs(DOCS_DIR, exist_ok=True)

# Check for Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

# ------------------------------
# Utility functions
# ------------------------------
def load_lottie_url(url):
    """Fetch Lottie animation from URL."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.warning(f"Failed to load Lottie animation: {e}")
    return None

# Animation URLs (fallback if not reachable)
ANIM_AI = load_lottie_url("https://lottie.host/9f7e7161-e009-4179-9944-77443f07a685/vIu8pU0f2o.json")
ANIM_SCAN = load_lottie_url("https://lottie.host/804a9d70-3481-4209-ab34-66c3a373b754/6fWpA7eY5t.json")
ANIM_CHAT = load_lottie_url("https://lottie.host/362a9816-160b-4682-960f-1a980868153c/p1S4tHqj1X.json")

# ------------------------------
# Cached resources
# ------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings(model_name=DEFAULT_EMBEDDING_MODEL):
    """Return cached HuggingFace embeddings (sentence-transformers)."""
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource(show_spinner="Loading Groq LLM...")
def get_llm(model_name=DEFAULT_GROQ_MODEL, temperature=DEFAULT_TEMPERATURE):
    """Return cached Groq chat model."""
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=GROQ_API_KEY,
    )

def get_vectorstore():
    """Load vector store from disk if exists."""
    if os.path.exists(INDEX_PATH):
        try:
            return FAISS.load_local(INDEX_PATH, get_embeddings(), allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            st.error("❌ Corrupted index. Please rebuild in Vault Manager.")
            # Delete corrupted index to allow fresh rebuild
            shutil.rmtree(INDEX_PATH, ignore_errors=True)
            return None
    return None

def rebuild_index(file_paths, chunk_size, chunk_overlap):
    """Rebuild FAISS index from uploaded files."""
    all_docs = []
    for path in file_paths:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".pdf":
                docs = PyPDFLoader(path).load()
            elif ext == ".docx":
                docs = Docx2txtLoader(path).load()
            elif ext == ".txt":
                docs = TextLoader(path).load()
            else:
                continue
            all_docs.extend(docs)
            logger.info(f"Loaded {Path(path).name} ({len(docs)} pages)")
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            st.error(f"⚠️ Error loading {Path(path).name}: {e}")

    if not all_docs:
        st.warning("No supported documents found.")
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(all_docs)
    logger.info(f"Split into {len(chunks)} chunks")

    # Remove old index if exists to avoid conflicts
    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH, ignore_errors=True)

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    vectorstore.save_local(INDEX_PATH)
    logger.info(f"Index saved to {INDEX_PATH}")
    return True

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="DocMind Pro (Groq + Sentence Transformers)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Custom CSS for pro feel
# ------------------------------
st.markdown("""
<style>
    /* Modern dark theme */
    .stApp {
        background: #0e1117;
        color: #e0e0e0;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1a1c24;
        border-right: 1px solid #2b313e;
    }
    /* Glassmorphism cards for chat */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        margin: 10px 0;
        padding: 10px;
    }
    /* Professional footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(10, 10, 15, 0.8);
        backdrop-filter: blur(10px);
        color: #888;
        text-align: center;
        padding: 8px;
        font-size: 12px;
        border-top: 1px solid #2b313e;
        z-index: 100;
    }
    /* Button hover effect */
    .stButton>button {
        border-radius: 20px;
        transition: all 0.3s ease;
        background: #2b313e;
        color: white;
        border: 1px solid #444;
    }
    .stButton>button:hover {
        border-color: #00d4ff;
        box-shadow: 0px 0px 15px #00d4ff55;
        transform: translateY(-2px);
    }
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00d4ff;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px;
        color: #888;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff;
        border-bottom: 2px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Initialize session state
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []  # track uploaded file paths for current session

# ------------------------------
# Sidebar navigation
# ------------------------------
with st.sidebar:
    if ANIM_AI:
        st_lottie(ANIM_AI, height=120, key="ai_anim")
    else:
        st.image("https://via.placeholder.com/300x120?text=DocMind", width=True)

    selected = option_menu(
        "DocMind",
        ["Dashboard", "Vault Manager", "Query Engine"],
        icons=["speedometer2", "archive", "chat-dots"],
        menu_icon="cpu",
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "nav-link-selected": {"background-color": "#00d4ff", "color": "#0e1117"},
        },
    )

    st.markdown("---")
    st.caption(f"🤖 **LLM:** {DEFAULT_GROQ_MODEL}")
    st.caption(f"📦 **Embedding:** {DEFAULT_EMBEDDING_MODEL.split('/')[-1]}")
    st.caption("⚡ **Status:** Online")

    # Advanced settings expander
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 300, 2000, DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, DEFAULT_CHUNK_OVERLAP, step=50)
        k_retrieval = st.slider("Retrieval Count (k)", 1, 15, DEFAULT_K_RETRIEVAL)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)

# ------------------------------
# Dashboard Page
# ------------------------------
if selected == "Dashboard":
    st.title("🧠 DocMind Pro (Groq + Sentence Transformers)")
    st.markdown("#### Enterprise‑grade RAG powered by Groq & local embeddings")

    col1, col2, col3 = st.columns(3)
    with col1:
        file_count = len([f for f in os.listdir(DOCS_DIR) if not f.startswith('.')])
        st.metric("Documents Stored", file_count)
    with col2:
        index_status = "✅ Active" if os.path.exists(INDEX_PATH) else "❌ Missing"
        st.metric("Index Status", index_status)
    with col3:
        st.metric("Context Window", "128K tokens (Groq)")

    st.markdown("---")
    if ANIM_CHAT:
        st_lottie(ANIM_CHAT, height=250)

# ------------------------------
# Vault Manager Page
# ------------------------------
elif selected == "Vault Manager":
    st.title("📂 Vault Manager")
    st.markdown("Upload documents and rebuild the knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Rebuild Index", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            # Clear old files and save new ones
            shutil.rmtree(DOCS_DIR, ignore_errors=True)
            os.makedirs(DOCS_DIR, exist_ok=True)

            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            with st.spinner("Indexing documents... This may take a moment."):
                success = rebuild_index(file_paths, chunk_size, chunk_overlap)
                if success:
                    st.success("✅ Index rebuilt successfully!")
                    st.balloons()
                else:
                    st.error("❌ Indexing failed. Check logs.")

    st.markdown("---")
    st.subheader("Current Vault Contents")
    files = [f for f in os.listdir(DOCS_DIR) if not f.startswith('.')]
    if files:
        for f in files:
            st.write(f"📄 {f}")
    else:
        st.info("Vault is empty. Upload files to begin.")

# ------------------------------
# Query Engine Page
# ------------------------------
elif selected == "Query Engine":
    st.title("💬 Query Engine")
    st.markdown("Ask questions based on your indexed documents.")

    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.warning("⚠️ No index found. Please upload documents in Vault Manager first.")
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrieval})
        llm = get_llm(temperature=temperature)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a precise AI assistant. Answer the question using ONLY the context provided. "
                "If the answer is not in the context, say 'I don't have that information.' "
                "Always mention the source file name when possible.\n\n"
                "Context: {context}"
            )),
            ("human", "{input}")
        ])

        combine_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, combine_chain)

        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if query := st.chat_input("Type your question here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Generate answer
            with st.chat_message("assistant"):
                if ANIM_SCAN:
                    st_lottie(ANIM_SCAN, height=80, key=f"scan_{len(st.session_state.messages)}")
                with st.spinner("Analyzing documents..."):
                    try:
                        response = rag_chain.invoke({"input": query})
                        answer = response["answer"]
                        st.markdown(answer)

                        # Show sources
                        if "context" in response and response["context"]:
                            sources = set()
                            for doc in response["context"]:
                                source = doc.metadata.get("source", "unknown")
                                sources.add(os.path.basename(source))
                            st.caption(f"📚 Sources: {', '.join(sources)}")

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"❌ Error: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

        # Option to clear chat
        if st.session_state.messages and st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
<div class="footer">
    DocMind Professional (Groq + Sentence Transformers) v3.4 | 2026 | Local embeddings · Groq API
</div>
""", unsafe_allow_html=True)
