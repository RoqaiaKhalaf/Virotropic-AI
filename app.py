import streamlit as st
import os
import sys
import uuid

# --- 1. تصفير البيئة لمنع تضارب المكتبات ---
if "pinecone" in sys.modules:
    del sys.modules["pinecone"]

# --- 2. إعداد المفاتيح (Secrets) ---
for key in ["GROQ_API_KEY", "PINECONE_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 3. إعدادات الصفحة ---
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

# --- 4. التنسيق البصري (CSS) - السايدبار واللوجو ---
st.markdown(f"""
<style>
    /* تلوين السايدبار بالدرجة المطلوبة */
    [data-testid="stSidebar"] {{
        background-color: #722f37;
    }}
    /* تلوين نصوص السايدبار بالأبيض لتكون واضحة */
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: #ffffff !important;
    }}
    /* تنسيق كارت الترحيب المبسط */
    .welcome-box {{
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        border-right: 5px solid #722f37;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }}
    /* تحسين شكل أزرار الجلسات */
    .stButton>button {{
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# --- 5. إدارة الجلسات (Sessions) ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.chat_sessions[initial_id] = {{"title": "Initial Session", "messages": []}}
    st.session_state.current_chat = initial_id

# --- 6. المحرك التقني (RAG) ---
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = PineconeVectorStore(
        index_name="virotropic1",
        embedding=embeddings,
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    return vector_db, llm

vector_store, llm = load_rag_system()

# --- 7. القائمة الجانبية (Sidebar) ---
with st.sidebar:
    # إضافة اللوجو (تأكدي من وجود ملف logo.png في نفس الفولدر أو استبداله برابط)
    try:
        st.image("logo.png", use_container_width=True) 
    except:
        st.markdown("### 🔬 VIROTROPIC LOGO")
    
    st.markdown("---")
    if st.button("➕ New Research Session", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {{"title": "New Session", "messages": []}}
        st.session_state.current_chat = new_id
        st.rerun()
    
    st.markdown("### 🕒 Recent Chats")
    # عرض قائمة الجلسات للحفاظ عليها
    for chat_id, chat_data in list(st.session_state.chat_sessions.items()):
        is_active = (chat_id == st.session_state.current_chat)
        if st.button(chat_data["title"], key=f"btn_{chat_id}", use_container_width=True, 
                     type="primary" if is_active else "secondary"):
            st.session_state.current_chat = chat_id
            st.rerun()

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Medical PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Indexing..."):
            # (نفس كود الرفع السابق الخاص بكِ)
            st.success("File Ready!")

# --- 8. منطقة العرض الرئيسية ---
current_session = st.session_state.chat_sessions[st.session_state.current_chat]

# رسالة ترحيب مبسطة
if not current_session["messages"]:
    st.markdown(f"""
    <div class="welcome-box">
        <h3 style="margin-top:0; color:#722f37;">ViroTropic AI 🔬</h3>
        <p>مرحباً بك في مساعدك البحثي المتخصص. يمكنك البدء بسؤال عن أمراض المناطق الحارة أو رفع ملفات PDF لتحليلها.</p>
    </div>
    """, unsafe_allow_html=True)

# عرض الرسائل السابقة
for msg in current_session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# إدخال السؤال
query = st.chat_input("Ask a medical question...")
if query:
    # تحديث اسم الجلسة بناءً على أول سؤال
    if current_session["title"] in ["Initial Session", "New Session"]:
        current_session["title"] = query[:20] + "..."
    
    current_session["messages"].append({{"role": "user", "content": query}})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Analyzing..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\\n\\n".join([d.page_content for d in docs])
        response = llm.invoke(f"Context: {{context}}\\n\\nQuestion: {{query}}")
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Sources"):
                for d in docs:
                    st.caption(f"📍 {{d.metadata.get('source', 'Reference')}}")
        
        current_session["messages"].append({{"role": "assistant", "content": response.content}})
