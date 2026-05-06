import streamlit as st
import os
import sys
import uuid

# --- 1. تنظيف الذاكرة وضمان استيراد المكتبات الصحيحة ---
if "pinecone" in sys.modules:
    del sys.modules["pinecone"]

# إعداد المفاتيح
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 2. إعدادات الصفحة ---
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

# --- 3. إدارة الجلسات (Sessions Logic) ---
if "chat_sessions" not in st.session_state:
    # إنشاء أول جلسة تلقائياً
    initial_id = str(uuid.uuid4())
    st.session_state.chat_sessions = {initial_id: {"title": "New Research Session", "messages": []}}
    st.session_state.current_chat = initial_id

# دالة لإنشاء شات جديد
def start_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {"title": "New Research Session", "messages": []}
    st.session_state.current_chat = new_id

# دالة للتبديل بين الشات
def switch_chat(chat_id):
    st.session_state.current_chat = chat_id

# --- 4. دالة الاستشهاد المطورة ---
def build_apa_citation(metadata):
    # محاولة جلب البيانات من الميتاداتا أو وضع قيم افتراضية
    source = metadata.get('source', 'Unknown File')
    filename = os.path.basename(str(source))
    author = metadata.get('author') or metadata.get('creator') or "Medical Expert"
    year = metadata.get('year') or "2024"
    title = metadata.get('title') or filename.replace('.pdf', '')
    page = metadata.get('page')
    
    citation = f"**{author} ({year}).** *{title}*"
    if page is not None:
        citation += f" (p. {int(page) + 1})"
    citation += f" | File: {filename}"
    return citation

# --- 5. التصميم (UI) ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; }
    .welcome-card { background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #722F37; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #722F37;'>🔬 ViroTropic AI</h1>", unsafe_allow_html=True)

# --- 6. المحرك التقني (RAG) ---
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

# --- 7. القائمة الجانبية (Sidebar) ---
with st.sidebar:
    st.title("📂 Conversations")
    st.button("➕ Start New Chat", on_click=start_new_chat)
    
    st.divider()
    # عرض قائمة المحادثات القديمة
    for chat_id, chat_data in st.session_state.chat_sessions.items():
        # تمييز الشات الحالي بلون مختلف
        style_type = "primary" if chat_id == st.session_state.current_chat else "secondary"
        st.button(chat_data["title"], key=f"btn_{chat_id}", on_click=switch_chat, args=(chat_id,), type=style_type)

    st.divider()
    st.markdown("### 📥 Upload Data")
    uploaded_file = st.file_uploader("Upload Medical PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Indexing into Pinecone..."):
            temp_path = f"temp_{uuid.uuid4()}.pdf"
            with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                loader = PyMuPDFLoader(temp_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)
                vector_store.add_documents(chunks)
                st.sidebar.success("✅ File Added!")
            except Exception as e: st.sidebar.error(f"Error: {e}")
            finally: 
                if os.path.exists(temp_path): os.remove(temp_path)

# --- 8. عرض المحادثة ---
current_chat_id = st.session_state.current_chat
current_session = st.session_state.chat_sessions[current_chat_id]

# رسالة الترحيب
if not current_session["messages"]:
    st.markdown("""
    <div class="welcome-card">
        <h3>Welcome to ViroTropic AI! 👋</h3>
        <p>I can help you analyze medical papers and surveillance data. Start by asking a question or uploading a file.</p>
    </div>
    """, unsafe_allow_html=True)

# عرض الرسائل السابقة
for msg in current_session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Sources"):
                for cit in msg["citations"]: st.caption(f"📍 {cit}")

# مدخلات المستخدم
query = st.chat_input("Ask about Tropical Medicine...")
if query:
    # تحديث عنوان الشات بناءً على أول سؤال
    if not current_session["messages"]:
        current_session["title"] = query[:30] + "..."

    # إضافة رسالة المستخدم
    current_session["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    # البحث والرد
    with st.spinner("Searching knowledge base..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {query}\n\nAnswer:")
        
        # بناء الاستشهادات
        citations = list(set([build_apa_citation(d.metadata) for d in docs]))
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            if citations:
                with st.expander("📎 Sources"):
                    for cit in citations: st.caption(f"📍 {cit}")

        # حفظ رد الموديل
        current_session["messages"].append({
            "role": "assistant", 
            "content": response.content, 
            "citations": citations
        })
