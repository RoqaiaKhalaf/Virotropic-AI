import streamlit as st
import os
import sys
import uuid

# 1. تنظيف الذاكرة وضمان استيراد المكتبات الصحيحة
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

# 2. إعدادات الصفحة والتصميم
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

# --- دالة الاستشهاد بتنسيق APA ---
def format_apa(metadata):
    author = metadata.get('author') or metadata.get('creator') or "Medical Researcher"
    year = metadata.get('year') or "2024"
    title = metadata.get('title') or "Tropical Medicine Study"
    source = os.path.basename(str(metadata.get('source', 'Unknown')))
    page = metadata.get('page')
    
    cite = f"**{author} ({year}).** *{title}*."
    if page:
        cite += f" (p. {int(page) + 1})."
    cite += f" [Source: {source}]"
    return cite

# --- إدارة الجلسات (Callbacks) ---
if "chat_sessions" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.chat_sessions = {initial_id: {"title": "Initial Session", "messages": []}}
    st.session_state.current_chat = initial_id

def start_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {"title": "New Research", "messages": []}
    st.session_state.current_chat = new_id

def switch_chat(chat_id):
    st.session_state.current_chat = chat_id

# --- التصميم CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #FAF4E8; }
.hero { text-align: center; padding: 1.5rem; border-bottom: 2px solid #722F37; margin-bottom: 1.5rem; }
.hero h1 { font-family: 'Playfair Display', serif; color: #722F37; font-size: 2.2rem; margin: 0; }
.welcome-box { background: white; padding: 20px; border-radius: 10px; border-left: 5px solid #722F37; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }
</style>
<div class="hero">
    <h1>🔬 ViroTropic AI</h1>
    <p style="color: #8B6B6E; font-weight: bold;">Intelligent Medical Research Assistant</p>
</div>
""", unsafe_allow_html=True)

# 3. المحرك التقني
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

# 4. القائمة الجانبية (Sidebar)
with st.sidebar:
    st.markdown("<h2 style='color: #722F37;'>💬 Conversations</h2>", unsafe_allow_html=True)
    st.button("➕ Start New Chat", on_click=start_new_chat, use_container_width=True)
    
    st.divider()
    for chat_id, chat_data in st.session_state.chat_sessions.items():
        is_active = (chat_id == st.session_state.current_chat)
        st.button(
            chat_data["title"], 
            key=f"btn_{chat_id}", 
            on_click=switch_chat, 
            args=(chat_id,),
            type="primary" if is_active else "secondary",
            use_container_width=True
        )

    st.divider()
    uploaded_file = st.file_uploader("Upload PDF to Library", type="pdf")
    if uploaded_file:
        with st.spinner("Indexing..."):
            temp_path = f"temp_{uuid.uuid4()}.pdf"
            with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                loader = PyMuPDFLoader(temp_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(data)
                vector_store.add_documents(chunks)
                st.success("✅ Added!")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

# 5. عرض المحادثة
current_chat_id = st.session_state.current_chat
session = st.session_state.chat_sessions[current_chat_id]

# --- رسالة الترحيب الثابتة (تظهر فقط لو الشات فاضي) ---
if not session["messages"]:
    st.markdown("""
    <div class="welcome-box">
        <h3 style="margin-top:0; color:#722F37;">Welcome to ViroTropic!</h3>
        <p>I am your AI assistant specialized in Tropical Medicine, so how can i help you today?</p>
    </div>
    """, unsafe_allow_html=True)

for msg in session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Academic Sources (APA)"):
                for cit in msg["citations"]: st.caption(f"📍 {cit}")

# 6. المدخلات والردود
query = st.chat_input("Enter your research question...")
if query:
    if not session["messages"]: session["title"] = query[:25] + "..."
    
    session["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.spinner("Analyzing research database..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {query}\n\nAnswer professionally:")
        
        # تجهيز الاستشهادات بأسلوب APA
        apa_citations = list(set([format_apa(d.metadata) for d in docs]))
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Academic Sources (APA)"):
                for cit in apa_citations: st.caption(f"📍 {cit}")

        session["messages"].append({
            "role": "assistant", 
            "content": response.content, 
            "citations": apa_citations
        })
