import streamlit as st
import os
import base64
import uuid

# 1. إعداد المفاتيح (Secrets) - لازم تكون قبل أي استيراد للمكتبات اللي بتستخدمها
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# 2. إعدادات الصفحة
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

# 3. نظام إدارة الجلسات
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chat_sessions[first_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat = first_id

# 4. التصميم CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #FAF4E8; color: #2D1B1E; }
.hero { display: flex; align-items: center; justify-content: center; flex-direction: column; padding: 2.5rem 0 1.8rem; border-bottom: 2px solid #722F37; margin-bottom: 2rem; text-align: center; }
.hero h1 { font-family: 'Playfair Display', serif; font-size: 2.6rem; color: #722F37; margin: 0; }
</style>
<div class="hero">
    <h1>🔬 ViroTropic AI</h1>
    <p style="color: #8B6B6E; font-weight: 700;">Intelligent Medical Research Assistant</p>
</div>
""", unsafe_allow_html=True)

# 5. منطق الاستشهاد
def build_apa_citation(metadata):
    author = metadata.get('author') or "Medical Expert"
    year = metadata.get('year') or "2024"
    source = metadata.get('source') or "Research Paper"
    return f"{author}. ({year}). *Study*. Source: {os.path.basename(str(source))}."

# 6. المحرك التقني
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    INDEX_NAME = "virotropic1"
    vector_db = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    return vector_db, llm

vector_store, llm = load_rag_system()

# 7. Sidebar (المكان اللي كان فيه خطأ المسافات)
with st.sidebar:
    st.markdown("<h2 style='color: #722F37;'>💬 Conversations</h2>", unsafe_allow_html=True)
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat = new_id
        st.rerun()

    for chat_id, chat_data in list(st.session_state.chat_sessions.items()):
        is_active = (chat_id == st.session_state.current_chat)
        if st.button(chat_data["title"], key=f"chat_{chat_id}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_chat = chat_id
            st.rerun()

    st.divider()
    st.markdown("<h2 style='color: #722F37;'>📂 Research Center</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Uploading..."):
            temp_path = f"temp_{uuid.uuid4()}.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                loader = PyMuPDFLoader(temp_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
                new_chunks = text_splitter.split_documents(data)
                vector_store.add_documents(new_chunks)
                st.success("✅ Done!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

# 8. عرض المحادثة والرد
current_session = st.session_state.chat_sessions[st.session_state.current_chat]
for msg in current_session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask me...")
if query:
    current_session["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.write(query)
    
    with st.spinner("Thinking..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Citations"):
                for d in docs: st.caption(build_apa_citation(d.metadata))
        
        current_session["messages"].append({"role": "assistant", "content": response.content})
