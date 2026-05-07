import streamlit as st
import os
import sys
import uuid
import base64

# 1. حل مشكلة التوافق وتحميل المفاتيح
if "pinecone" in sys.modules:
    del sys.modules["pinecone"]

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# 2. إعدادات الصفحة
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

# 3. نظام إدارة الجلسات المتقدم
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

# 4. معالجة اللوجو (Base64 لضمان الظهور)
def get_logo_base64(path="logo.png"):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except: return None

logo_src = get_logo_base64()
logo_html = f'<img src="{logo_src}" style="height:52px; vertical-align:middle; margin-right:12px;">' if logo_src else "🔬"

# 5. التصميم CSS المتطور (المطلوب)
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: #FAF4E8; color: #2D1B1E; }}
.hero {{ display: flex; align-items: center; justify-content: center; flex-direction: column; padding: 2.5rem 0 1.8rem; border-bottom: 2px solid #722F37; margin-bottom: 2rem; text-align: center; }}
.hero h1 {{ font-family: 'Playfair Display', serif; font-size: 2.6rem; color: #722F37; margin: 0; display: inline; vertical-align: middle; }}
.hero p {{ color: #8B6B6E; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 0.5rem; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {{ background: #722F37; border-radius: 16px 16px 4px 16px; padding: 0.9rem 1.2rem; color: #FAF4E8; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {{ background: #FFFFFF; border: 1px solid #E8DECE; border-left: 3px solid #722F37; border-radius: 4px 16px 16px 16px; padding: 0.9rem 1.2rem; color: #2D1B1E; }}
button[kind="primary"] {{ background-color: #5C3D3D !important; color: #FAF4E8 !important; border: none !important; }}
</style>
<div class="hero">
    <div>{logo_html} <h1>ViroTropic</h1></div>
    <p>Intelligent Medical Research Assistant</p>
</div>
""", unsafe_allow_html=True)

# 6. منطق الاستشهاد APA (المحسن)
def build_apa_citation(metadata):
    author = metadata.get('author') or metadata.get('creator') or "Medical Researcher"
    year = metadata.get('year') or "2024"
    title = metadata.get('title') or "Tropical Medicine Study"
    source = os.path.basename(str(metadata.get('source', 'Unknown')))
    page = metadata.get('page')
    cite = f"**{author} ({year}).** *{title}*."
    if page: cite += f" (p. {int(page) + 1})."
    cite += f" [Source: {source}]"
    return cite

# 7. المحرك التقني (Pinecone + Groq)
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

# 8. القائمة الجانبية (إدارة المحادثات المتقدمة)
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
                new_name = st.text_input("New Name:", value=chat_data["title"], key=f"in_{chat_id}")
                if st.button("✅ Save", key=f"sv_{chat_id}"):
                    st.session_state.chat_sessions[chat_id]["title"] = new_name
                    st.session_state.renaming_chat = None
                    st.session_state.open_menu = None
                    st.rerun()
            else:
                c1, c2 = st.columns(2)
                if c1.button("✏️", key=f"ren_{chat_id}"): 
                    st.session_state.renaming_chat = chat_id
                    st.rerun()
                if c2.button("🗑️", key=f"del_{chat_id}"):
                    del st.session_state.chat_sessions[chat_id]
                    if not st.session_state.chat_sessions:
                        nid = str(uuid.uuid4())
                        st.session_state.chat_sessions[nid] = {"title": "New Chat", "messages": []}
                        st.session_state.current_chat = nid
                    elif st.session_state.current_chat == chat_id:
                        st.session_state.current_chat = list(st.session_state.chat_sessions.keys())[0]
                    st.rerun()

    st.divider()
    st.markdown("<h2 style='color: #722F37;'>📂 Research Center</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Indexing..."):
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
                st.success("✅ Document Integrated!")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

# 9. منطقة المحادثة
current_session = st.session_state.chat_sessions[st.session_state.current_chat]
if not current_session["messages"]:
    welcome_msg = "<div style='font-weight:bold; color:#722F37;'>Welcome to ViroTropic AI.</div> Ready for research?"
    current_session["messages"].append({"role": "assistant", "content": welcome_msg, "citations": []})

for m in current_session["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)
        if m.get("citations"):
            with st.expander("📎 Academic Citations (APA)"):
                for c in m["citations"]: st.caption(f"📍 {c}")

query = st.chat_input("Ask a medical question...")
if query:
    if current_session["title"] == "New Chat": current_session["title"] = query[:30] + "..."
    current_session["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.write(query)

    with st.spinner("Searching Archives..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {query}\n\nAnswer professionally:")
        
        cits = list(set([build_apa_citation(d.metadata) for d in docs]))
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Academic Citations (APA)"):
                for c in cits: st.caption(f"📍 {c}")

        current_session["messages"].append({"role": "assistant", "content": response.content, "citations": cits})
