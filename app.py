import streamlit as st
import os
import sys
import uuid
import base64

# --- 1. تنظيف الذاكرة ومنع التضارب ---
if "pinecone" in sys.modules:
    del sys.modules["pinecone"]

# --- 2. إعداد البيئة (Secrets) ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 3. دالة معالجة اللوجو (Logo) ---
# تأكدي من وجود ملف باسم logo.png في نفس فولدر المشروع على GitHub
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_logo(main_bg_html):
    try:
        bin_str = get_base64_of_bin_file('logo.png')
        return f'<img src="data:image/png;base64,{bin_str}" class="main-logo">'
    except:
        return "🔬" # ايموجي احتياطي لو الصورة مش موجودة

# --- 4. إعدادات الصفحة والتصميم CSS المخصص ---
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

main_color = "#722f37"
secondary_bg = "#fdfaf5"

st.markdown(f"""
<style>
    /* تنسيق السايدبار باللون المطلوب */
    [data-testid="stSidebar"] {{
        background-color: {main_color};
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    /* تنسيق الخطوط في الصفحة الأساسية */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: #2D1B1E;
    }}
    
    .main-logo {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 120px;
        margin-bottom: 10px;
    }}
    
    .welcome-card {{
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        border-right: 4px solid {main_color};
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }}
    
    /* تنسيق أزرار السايدبار */
    .stButton>button {{
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.3);
    }}
</style>
""", unsafe_allow_html=True)

# --- 5. نظام إدارة الجلسات ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chat_sessions[first_id] = {"title": "Initial Session", "messages": []}
    st.session_state.current_chat = first_id

# --- 6. عرض الهيدر واللوجو ---
logo_html = set_logo("")
st.markdown(f"""
    <div style="text-align: center;">
        {logo_html}
        <h1 style="color: {main_color}; margin-top: 0;">ViroTropic AI</h1>
        <p style="color: #666; font-style: italic;">Empowering Tropical Medicine Research</p>
    </div>
""", unsafe_allow_html=True)

# رسالة الترحيب الثابتة
current_session = st.session_state.chat_sessions[st.session_state.current_chat]
if not current_session["messages"]:
    st.markdown(f"""
    <div class="welcome-card">
        <h3 style="color: {main_color}; margin-top: 0;">Welcome, Researcher! 👋</h3>
        <p>Your specialized AI environment is ready. You can start by asking about infectious diseases or uploading your latest PDF studies.</p>
    </div>
    """, unsafe_allow_html=True)

# --- 7. المحرك التقني (RAG) ---
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

# --- 8. السايدبار الملون ---
with st.sidebar:
    st.markdown("### 🛠️ Control Center")
    if st.button("➕ New Research Session", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {"title": "New Research", "messages": []}
        st.session_state.current_chat = new_id
        st.rerun()
    
    st.divider()
    st.markdown("### 📂 Medical Library")
    uploaded_file = st.file_uploader("Upload Study (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Indexing..."):
            # نفس كود الرفع السابق
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
                st.success("✅ Added to Library")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

# --- 9. منطقة المحادثة ---
for msg in current_session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Academic Citations"):
                for cit in msg["citations"]:
                    st.caption(f"📍 {cit}")

query = st.chat_input("Ask ViroTropic AI...")
if query:
    if current_session["title"] == "Initial Session":
        current_session["title"] = query[:20] + "..."
    
    current_session["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.spinner("Analyzing Database..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer professionally:"
        response = llm.invoke(prompt)
        
        # استخراج استشهادات (باستخدام الدالة الأكاديمية اللي عملناها)
        citations = list(set([f"Source: {os.path.basename(str(d.metadata.get('source', 'Doc')))}" for d in docs]))
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Academic Citations"):
                for cit in citations: st.caption(f"📍 {cit}")
        
        current_session["messages"].append({"role": "assistant", "content": response.content, "citations": citations})
