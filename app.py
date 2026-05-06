import streamlit as st
import os
import sys
import uuid

# 1. حيلة تقنية لإجبار السيرفر على نسيان أي نسخة قديمة من Pinecone
if "pinecone" in sys.modules:
    del sys.modules["pinecone"]

# 2. إعداد المفاتيح من الـ Secrets أولاً
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# 3. الآن نقوم بالاستيراد (Import)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# 4. إعدادات الصفحة والتصميم
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #FAF4E8; }
.hero { text-align: center; padding: 2rem; border-bottom: 2px solid #722F37; margin-bottom: 2rem; }
.hero h1 { font-family: 'Playfair Display', serif; color: #722F37; font-size: 2.5rem; }
</style>
<div class="hero">
    <h1>🔬 ViroTropic AI</h1>
    <p style="color: #8B6B6E;">Intelligent Medical Research Assistant</p>
</div>
""", unsafe_allow_html=True)

# 5. إدارة جلسات الدردشة
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chat_sessions[first_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat = first_id

# 6. المحرك التقني (RAG System)
@st.cache_resource
def load_rag_system():
    # تأكدي أن هذا الموديل متوافق مع البيانات المرفوعة
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    INDEX_NAME = "virotropic1"
    
    vector_db = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    return vector_db, llm

# محاولة تشغيل النظام
try:
    vector_store, llm = load_rag_system()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# 7. القائمة الجانبية (Sidebar)
with st.sidebar:
    st.markdown("<h2 style='color: #722F37;'>💬 Conversations</h2>", unsafe_allow_html=True)
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat = new_id
        st.rerun()

    for chat_id, chat_data in list(st.session_state.chat_sessions.items()):
        is_active = (chat_id == st.session_state.current_chat)
        if st.button(chat_data["title"], key=f"chat_{chat_id}", use_container_width=True, 
                     type="primary" if is_active else "secondary"):
            st.session_state.current_chat = chat_id
            st.rerun()

    st.divider()
    st.markdown("<h3 style='color: #722F37;'>📂 Upload Research</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            temp_path = f"temp_{uuid.uuid4()}.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                loader = PyMuPDFLoader(temp_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
                chunks = text_splitter.split_documents(data)
                vector_store.add_documents(chunks)
                st.success("✅ Indexed Successfully!")
            except Exception as e:
                st.error(f"Upload Error: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

# 8. منطقة الدردشة
current_session = st.session_state.chat_sessions[st.session_state.current_chat]
for msg in current_session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about Tropical Medicine...")
if query:
    if current_session["title"] == "New Chat":
        current_session["title"] = query[:25] + "..."
        
    current_session["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.spinner("Analyzing research..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer like a medical professional:"
        response = llm.invoke(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Sources"):
                for d in docs:
                    st.caption(f"📍 From: {os.path.basename(str(d.metadata.get('source', 'Unknown')))}")

        current_session["messages"].append({"role": "assistant", "content": response.content})
