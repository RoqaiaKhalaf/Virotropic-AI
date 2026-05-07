import streamlit as st
import os
import sys
import uuid
import base64

# ── 1. إعداد المكتبات والأدوات (Pinecone & Groq) ──────────────────────
if "pinecone" in sys.modules:
    del sys.modules["pinecone"]

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# ── 2. إعدادات الصفحة ─────────────────────────────────────────────────────
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered", initial_sidebar_state="expanded")

# ── 3. نظام إدارة الجلسات والمنيو ─────────────────────────────────────────
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

# ── 4. تحميل اللوجو ───────────────────────────────────────────────────────
def get_logo_base64(path="logo.png"):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return None

logo_src = get_logo_base64()
logo_html = f'<img src="{logo_src}" style="height:52px; vertical-align:middle; margin-right:12px;">' if logo_src else "🔬"

# ── 5. التصميم CSS (نفس الستايل المطور) ──────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {{ 
    font-family: 'Inter', sans-serif; 
    background-color: #FAF4E8; 
    color: #2D1B1E; 
}}

.hero {{ display: flex; align-items: center; justify-content: center; flex-direction: column; padding: 2.5rem 0 1.8rem; border-bottom: 2px solid #722F37; margin-bottom: 2rem; text-align: center; }}
.hero h1 {{ font-family: 'Playfair Display', serif; font-size: 2.6rem; color: #722F37; margin: 0; display: inline; vertical-align: middle; }}
.hero p {{ color: #8B6B6E; font-size: 0.9rem; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 0.5rem; }}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {{ background: #722F37; border-radius: 16px 16px 4px 16px; padding: 0.9rem 1.2rem; color: #FAF4E8; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {{ background: #FFFFFF; border: 1px solid #E8DECE; border-left: 3px solid #722F37; border-radius: 4px 16px 16px 16px; padding: 0.9rem 1.2rem; color: #2D1B1E; }}

button[kind="primary"] {{
    background-color: #5C3D3D !important;
    border-color: #5C3D3D !important;
    color: #FAF4E8 !important;
}}
</style>

<div class="hero">
    <div class="hero-top"> {logo_html} <h1>ViroTropic</h1> </div>
    <p>Intelligent Medical Research Assistant</p>
</div>
""", unsafe_allow_html=True)

# ── 6. منطق الاستشهاد APA ──────────────────────────────────────────────
def build_apa_citation(metadata):
    author = metadata.get('author') or metadata.get('creator') or "Unknown Author"
    year = metadata.get('year') or "2024"
    title = metadata.get('title') or "Tropical Medicine Study"
    source = os.path.basename(str(metadata.get('source', 'Ref')))
    page = metadata.get('page')
    
    cite = f"**{author} ({year}).** *{title}*."
    if page: cite += f" (p. {int(page) + 1})."
    cite += f" [Source: {source}]"
    return cite

# ── 7. المحرك التقني (Pinecone & Groq) ──────────────────────────────────
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

# ── 8. القائمة الجانبية (Conversations Management) ───────────────────────
with st.sidebar:
    st.markdown("<h2 style='color: #722F37;'>💬 Conversations</h2>", unsafe_allow_html=True)
    
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat = new_id
        st.rerun()

    for chat_id, chat_data in list(st.session_state.chat_sessions.items()):
        is_active = (chat_id == st.session_state.current_chat)
        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(chat_data["title"], key=f"chat_{chat_id}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.current_chat = chat_id
                st.rerun()
        with col2:
            if st.button("⋯", key=f"menu_{chat_id}"):
                st.session_state.open_menu = chat_id if st.session_state.open_menu != chat_id else None
                st.rerun()

        if st.session_state.open_menu == chat_id:
            if st.session_state.renaming_chat == chat_id:
                new_name = st.text_input("New name:", value=chat_data["title"], key=f"ren_{chat_id}")
                c1, c2 = st.columns(2)
                if c1.button("✅ Done", key=f"save_{chat_id}"):
                    if new_name.strip(): st.session_state.chat_sessions[chat_id]["title"] = new_name.strip()
                    st.session_state.renaming_chat = None
                    st.session_state.open_menu = None
                    st.rerun()
                if c2.button("✖ close", key=f"can_{chat_id}"):
                    st.session_state.renaming_chat = None
                    st.rerun()
            else:
                m1, m2 = st.columns(2)
                if m1.button("✏️ Rename", key=f"ed_{chat_id}"):
                    st.session_state.renaming_chat = chat_id
                    st.rerun()
                if m2.button("🗑️ Delete", key=f"del_{chat_id}"):
                    del st.session_state.chat_sessions[chat_id]
                    if st.session_state.current_chat == chat_id:
                        st.session_state.current_chat = list(st.session_state.chat_sessions.keys())[0] if st.session_state.chat_sessions else None
                    st.rerun()

    st.divider()
    st.markdown("<h3 style='color: #722F37;'>📂 Research Center</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload new papers to expand my knowledge base", type="pdf")
    if uploaded_file:
        with st.spinner("Indexing..."):
            temp_path = f"temp_{uuid.uuid4()}.pdf"
            with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                loader = PyMuPDFLoader(temp_path)
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
                vector_store.add_documents(chunks)
                st.success("✅ Document Integrated")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

# ── 9. عرض المحادثة ─────────────────────────────────────────────────────
current_session = st.session_state.chat_sessions[st.session_state.current_chat]
current_messages = current_session["messages"]

if not current_messages:
    # الجزء العلوي: الأيقونة والعنوان
    st.markdown(f"""
    <div style="text-align: center; padding-top: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 10px;">🔬</div>
        <h2 style="color: #722F37; font-family: 'Playfair Display', serif; font-size: 2rem; margin-bottom: 2rem;">
            How can I help you today?
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # صف البطاقات (3 أعمدة كما في الصورة)
    col1, col2, col3 = st.columns(3)

    card_style = """
        background-color: #FFFFFF;
        border: 1px solid #E8DECE;
        border-radius: 12px;
        padding: 15px;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s;
    """

    with col1:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size: 1.2rem; margin-bottom: 5px;">📝</div>
            <div style="font-size: 0.85rem; color: #2D1B1E; font-weight: 500;">
                Summarize research papers about Malaria outbreaks
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size: 1.2rem; margin-bottom: 5px;">💡</div>
            <div style="font-size: 0.85rem; color: #2D1B1E; font-weight: 500;">
                Explain the latest trends in Tropical Medicine
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size: 1.2rem; margin-bottom: 5px;">📜</div>
            <div style="font-size: 0.85rem; color: #2D1B1E; font-weight: 500;">
                Generate APA citations for my medical documents
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 3rem;'></div>", unsafe_allow_html=True)

# عرض الرسائل القديمة كما هي
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("citations"):
            with st.expander("📎 Academic Citations (APA)"):
                for cit in message["citations"]: st.caption(f"📍 {cit}")

# ── 10. الإدخال والمعالجة ────────────────────────────────────────────────
query = st.chat_input("Ask about Tropical disease, Malaria or Research paper...")

if query:
    if current_session["title"] == "New Chat":
        current_session["title"] = query[:30] + "..."
    
    current_messages.append({"role": "user", "content": query, "citations": []})
    with st.chat_message("user"): st.write(query)

    with st.spinner("Searching ViroTropic Archives..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {query}\n\nAnswer professionally:")
        
        apa_citations = list(set([build_apa_citation(d.metadata) for d in docs]))
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Academic Citations (APA)"):
                for cit in apa_citations: st.caption(f"📍 {cit}")
        
        current_messages.append({"role": "assistant", "content": response.content, "citations": apa_citations})
