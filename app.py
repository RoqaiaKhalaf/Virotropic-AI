import streamlit as st
import os
import sys
import uuid
from PIL import Image

# --- 1. تنظيف الذاكرة ---
if "pinecone" in sys.modules:
    del sys.modules["pinecone"]

# --- 2. إعداد المفاتيح ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 3. دالة الاستشهاد ---
def build_apa_citation(metadata):
    source_path = metadata.get('source', 'Unknown Document')
    filename = os.path.basename(str(source_path))
    author = metadata.get('author') or metadata.get('creator') or "Medical Research Team"
    year = metadata.get('year') or "2024"
    title = metadata.get('title') or filename.replace('.pdf', '')
    page = metadata.get('page', None)
    citation = f"**{author} ({year}).** *{title}*"
    if page is not None:
        citation += f" (p. {int(page) + 1})"
    return f"{citation} | Source: {filename}"

# --- 4. إعدادات الصفحة ---
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

# --- 5. إضافة اللوجو في الهيدر (Header) ---
# تأكدي من تسمية ملف اللوجو logo.png ورفعه بجانب app.py على GitHub
col1, col2 = st.columns([1, 4])
with col1:
    try:
        logo = Image.open("logo.png") # تأكدي من اسم الملف
        st.image(logo, width=120)
    except:
        st.info("Logo not found") # لو الصورة مش موجودة هيظهر نص بسيط

with col2:
    st.markdown("<h1 style='color: #722F37; margin-top: 10px;'>ViroTropic AI</h1>", unsafe_allow_html=True)

# --- 6. رسالة الترحيب الثابتة ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chat_sessions[first_id] = {"title": "Initial Session", "messages": []}
    st.session_state.current_chat = first_id

current_session = st.session_state.chat_sessions[st.session_state.current_chat]
if not current_session["messages"]:
    st.markdown("""
    <div style="background-color: #FFFFFF; border-radius: 15px; padding: 25px; border-left: 5px solid #722F37; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <h2 style="color: #722F37; margin-top: 0;">Welcome to ViroTropic 👋</h2>
        <p>I am your specialized AI for Tropical Medicine and Infectious Diseases. I can help you analyze medical research like the study on <b>Dengue surveillance</b>.</p>
    </div>
    """, unsafe_allow_html=True)

# --- 7. المحرك التقني ---
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = PineconeVectorStore(
        index_name="virotropic1",
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    return vector_db, llm

vector_store, llm = load_rag_system()

# --- 8. القائمة الجانبية (Sidebar) مع لوجو إضافي ---
with st.sidebar:
    try:
        st.image("logo.png", use_column_width=True) # لوجو في الـ Sidebar برضه
    except:
        pass
    st.title("📂 Control Panel")
    if st.button("➕ Start New Session", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {"title": "New Research", "messages": []}
        st.session_state.current_chat = new_id
        st.rerun()
    
    st.divider()
    uploaded_file = st.file_uploader("Library Upload (PDF)", type="pdf")
    # ... (كود الرفع يوضع هنا) ...

# --- 9. الدردشة ---
for msg in current_session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Academic Citations"):
                for cit in msg["citations"]: st.caption(f"📍 {cit}")

query = st.chat_input("Enter your inquiry...")
if query:
    current_session["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    
    with st.spinner("Analyzing..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
        citations = list(set([build_apa_citation(d.metadata) for d in docs]))
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Academic Citations"):
                for cit in citations: st.caption(f"📍 {cit}")
        
        current_session["messages"].append({"role": "assistant", "content": response.content, "citations": citations})
