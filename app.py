import streamlit as st
import os
import sys
import uuid

# --- 1. إصلاح تضارب المكتبات وحمل المفاتيح ---
if "pinecone" in sys.modules:
    del sys.modules["pinecone"]

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 2. إعدادات الصفحة ---
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

# --- 3. نظام الـ Sessions (إصلاح مشكلة الشات الجديد) ---
if "chat_sessions" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.chat_sessions = {initial_id: {"title": "Initial Session", "messages": []}}
    st.session_state.current_chat = initial_id

# دوال التحكم في الشات (Callbacks)
def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {"title": "New Research", "messages": []}
    st.session_state.current_chat = new_id

def switch_to_chat(chat_id):
    st.session_state.current_chat = chat_id

# --- 4. دالة الاستشهاد (APA Style) ---
def build_apa_citation(metadata):
    # استخراج البيانات أو وضع قيم افتراضية ذكية
    source_path = metadata.get('source', 'Unknown')
    filename = os.path.basename(str(source_path))
    
    author = metadata.get('author') or "Medical Expert"
    # استخراج السنة من الـ metadata أو من اسم الملف لو موجودة
    year = metadata.get('year') or "2024"
    title = metadata.get('title') or filename.split('.')[0]
    page = metadata.get('page')

    # تنسيق APA: Author (Year). Title.
    citation = f"**{author} ({year}).** *{title}*."
    if page is not None:
        citation += f" (Page {int(page) + 1})."
    
    citation += f" Source File: {filename}"
    return citation

# --- 5. الواجهة (UI) ---
st.markdown("<h1 style='text-align: center; color: #722F37;'>🔬 ViroTropic AI</h1>", unsafe_allow_html=True)

# --- 6. المحرك التقني ---
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

# --- 7. القائمة الجانبية (إصلاح الحفظ والتنقل) ---
with st.sidebar:
    st.markdown("### 💬 Chat Management")
    # زرار الشات الجديد لازم يستخدم on_click
    st.button("➕ Start New Research", on_click=create_new_chat, use_container_width=True)
    
    st.divider()
    st.markdown("### 🕒 History")
    for chat_id, chat_data in st.session_state.chat_sessions.items():
        is_active = (chat_id == st.session_state.current_chat)
        # زرار التبديل بين المحادثات
        st.button(
            chat_data["title"], 
            key=f"btn_{chat_id}", 
            on_click=switch_to_chat, 
            args=(chat_id,),
            type="primary" if is_active else "secondary",
            use_container_width=True
        )

    st.divider()
    uploaded_file = st.sidebar.file_uploader("Upload to Library", type="pdf")
    if uploaded_file:
        # كود الرفع الخاص بكِ كما هو...
        pass

# --- 8. منطقة عرض المحادثة ---
current_chat = st.session_state.chat_sessions[st.session_state.current_chat]

# رسالة ترحيب لو الشات فاضي
if not current_chat["messages"]:
    st.info("👋 Welcome! Ready to analyze your tropical medicine documents.")

for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Grounding Sources (APA Style)"):
                for cit in msg["citations"]:
                    st.markdown(f"- {cit}")

# --- 9. منطقة الإدخال ---
query = st.chat_input("Ask a medical question...")

if query:
    # تحديث عنوان الشات بأول سؤال
    if not current_chat["messages"]:
        current_chat["title"] = query[:25] + "..."

    current_chat["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.spinner("Searching..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
        
        # تحويل الميتاداتا الخام لاستشهادات APA
        apa_citations = list(set([build_apa_citation(d.metadata) for d in docs]))
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Grounding Sources (APA Style)"):
                for cit in apa_citations:
                    st.markdown(f"- {cit}")

        current_chat["messages"].append({
            "role": "assistant", 
            "content": response.content,
            "citations": apa_citations
        })
