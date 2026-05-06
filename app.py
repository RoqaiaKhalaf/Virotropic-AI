import streamlit as st
import os
import base64
import uuid

# ── 1. التعامل مع الـ Secrets والمكتبات ──────────────────────────────────
# تأكدي من وضع المفاتيح في Streamlit Cloud Secrets أو ملف .env
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# ── 2. إعدادات الصفحة ─────────────────────────────────────────────────────
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

# ── 3. نظام إدارة الجلسات (نفس التصميم الجديد) ──────────────────────────────
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chat_sessions[first_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat = first_id

if "open_menu" not in st.session_state:
    st.session_state.open_menu = None
if "renaming_chat" not in st.session_state:
    st.session_state.renaming_chat = None

# ── 4. تحميل اللوجو (تعديل المسار ليعمل على السيرفر) ──────────────────────────
def get_logo_base64(path="logo.png"): # افترضنا أن اللوجو في نفس مجلد الكود على GitHub
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except FileNotFoundError:
        return None

logo_src = get_logo_base64()
logo_html = f'<img src="{logo_src}" style="height:52px; vertical-align:middle; margin-right:12px;">' if logo_src else "🔬"

# ── 5. التصميم CSS (كما في الكود الجديد تماماً) ──────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: #FAF4E8; color: #2D1B1E; }}
.hero {{ display: flex; align-items: center; justify-content: center; flex-direction: column; padding: 2.5rem 0 1.8rem; border-bottom: 2px solid #722F37; margin-bottom: 2rem; text-align: center; }}
.hero h1 {{ font-family: 'Playfair Display', serif; font-size: 2.6rem; color: #722F37; margin: 0; }}
.hero p {{ color: #8B6B6E; font-size: 0.9rem; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 0.5rem; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {{ background: #722F37; border-radius: 16px 16px 4px 16px; padding: 0.9rem 1.2rem; color: #FAF4E8; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {{ background: #FFFFFF; border: 1px solid #E8DECE; border-left: 3px solid #722F37; border-radius: 4px 16px 16px 16px; padding: 0.9rem 1.2rem; color: #2D1B1E; }}
button[kind="primary"] {{ background-color: #5C3D3D !important; border-color: #5C3D3D !important; color: #FAF4E8 !important; }}
</style>
<div class="hero">
    <div class="hero-top"> {logo_html} <h1>ViroTropic</h1> </div>
    <p>Intelligent Medical Research Assistant</p>
</div>
""", unsafe_allow_html=True)

# ── 6. منطق الاستشهاد (APA Citation) ──────────────────────────────────────
def build_apa_citation(metadata, doc_content):
    author = metadata.get('author') or "Unknown Author"
    year = metadata.get('year') or "n.d."
    title = metadata.get('title') or f"Document_{os.path.basename(metadata.get('source', 'Ref'))}"
    repo = "Liverpool School of Tropical Medicine Repository"
    return f"{author}. ({year}). *{title}*. {repo}."

# ── 7. المحرك التقني (Groq + Pinecone) ───────────────────────────────────
@st.cache_resource
def load_rag_system():
    # استخدام الموديل المتوافق مع بياناتك المرفوعة
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # الربط بـ Pinecone
    INDEX_NAME = "virotropic1"
    vector_db = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )
    
    # الربط بـ Groq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    return vector_db, llm

vector_store, llm = load_rag_system()

# ── 8. القائمة الجانبية (Sidebar) ──────────────────────────────────────────
# (تم الاحتفاظ بكل ميزات الـ Rename والـ Delete من التصميم الجديد)
with st.sidebar:
    st.markdown("<h2 style='color: #722F37;'>💬 Conversations</h2>", unsafe_allow_html=True)
    
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat = new_id
        st.rerun()

    for chat_id, chat_data in list(st.session_state.chat_sessions.items()):
        is_active = chat_id == st.session_state.current_chat
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(chat_data["title"], key=f"chat_{chat_id}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.current_chat = chat_id
                st.rerun()
        # ... (باقي كود الـ Rename/Delete كما هو في الكود الذي أرفقتِه) ...

# ── 9. نظام المحادثة (الربط مع Groq) ──────────────────────────────────────
current_session = st.session_state.chat_sessions[st.session_state.current_chat]
current_messages = current_session["messages"]

# عرض الرسائل السابقة
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("citations"):
            with st.expander("📎 Academic Citations"):
                for cit in message["citations"]: st.caption(cit)

query = st.chat_input("Ask about Tropical Diseases...")

if query:
    # تحديث عنوان الشات تلقائياً
    if current_session["title"] == "New Chat":
        current_session["title"] = query[:30] + "..."
    
    current_messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.write(query)

    with st.spinner("Searching ViroTropic Archives..."):
        # 1. البحث في Pinecone
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        # 2. إرسال لـ Groq
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer as a professional medical researcher:"
        response = llm.invoke(prompt)
        
        # 3. تجهيز الاستشهادات
        current_citations = [build_apa_citation(d.metadata, d.page_content) for d in docs]
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Academic Citations"):
                for cit in current_citations: st.caption(cit)
        
        current_messages.append({
            "role": "assistant", 
            "content": response.content, 
            "citations": current_citations
        })
