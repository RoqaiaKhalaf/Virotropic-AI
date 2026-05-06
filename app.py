import streamlit as st
import os
import sys
import uuid
import base64

# --- 1. حيلة تقنية لتنظيف الذاكرة ومنع تعارض المكتبات ---
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

# --- 3. دالة معالجة اللوجو (لتحويل الصورة لـ Base64 وعرضها في الـ CSS) ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def build_markup_for_logo(png_file):
    try:
        binary_string = get_base64_of_bin_file(png_file)
        return f"""
            <div style="position: absolute; top: -50px; right: -20px;">
                <img src="data:image/png;base64,{binary_string}" width="100">
            </div>
        """
    except:
        return "" # في حال لم يجد ملف الصورة لا يظهر خطأ

# --- 4. دالة الاستشهاد الأكاديمية ---
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
    citation += f" | Source: {filename}"
    return citation

# --- 5. إعدادات الصفحة ---
st.set_page_config(page_title="ViroTropic AI", page_icon="🔬", layout="centered")

# --- 6. نظام إدارة الجلسات ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chat_sessions[first_id] = {"title": "Initial Session", "messages": []}
    st.session_state.current_chat = first_id

# --- 7. الواجهة والتصميم (CSS) ---
# تأكدي أن ملف logo.png موجود في نفس المجلد
logo_html = build_markup_for_logo("logo.png") 

st.markdown(f"""
<style>
    .welcome-card {{
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 25px;
        border-left: 5px solid #722F37;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 30px;
    }}
    .welcome-card h2 {{ color: #722F37; margin-top: 0; }}
</style>
{logo_html}
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #722F37;'>🔬 ViroTropic AI</h1>", unsafe_allow_html=True)

# رسالة الترحيب الثابتة
current_session = st.session_state.chat_sessions[st.session_state.current_chat]
if not current_session["messages"]:
    st.markdown("""
    <div class="welcome-card">
        <h2>Welcome to ViroTropic Research Assistant! 👋</h2>
        <p>I am your specialized AI for Tropical Medicine and Infectious Diseases. How can I assist your research today?</p>
        <ul style="color: #555;">
            <li><b>Analyze Research Papers:</b> Upload clinical studies for instant summaries.</li>
            <li><b>Disease Surveillance:</b> Ask about outbreak prediction and monitoring tools.</li>
            <li><b>Academic Citation:</b> Get referenced answers directly from your library.</li>
        </ul>
        <p style="font-size: 0.85rem; color: #8B6B6E;"><i>Note: Please upload your PDF documents in the sidebar to begin.</i></p>
    </div>
    """, unsafe_allow_html=True)

# --- 8. المحرك التقني ---
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

try:
    vector_store, llm = load_rag_system()
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# --- 9. القائمة الجانبية (Sidebar) ---
with st.sidebar:
    st.title("📂 Control Panel")
    if st.button("➕ Start New Research Session", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {"title": "New Research", "messages": []}
        st.session_state.current_chat = new_id
        st.rerun()
    
    st.divider()
    uploaded_file = st.file_uploader("Add to Medical Library (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Indexing File..."):
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
                st.success("✅ Successfully added to Pinecone!")
            except Exception as e:
                st.error(f"Upload error: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

# --- 10. عرض الرسائل والردود ---
for msg in current_session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Grounding Sources (APA)"):
                for cit in msg["citations"]:
                    st.caption(f"📍 {cit}")

query = st.chat_input("Enter your medical inquiry...")
if query:
    if current_session["title"] == "Initial Session":
        current_session["title"] = query[:25] + "..."
    
    current_messages = current_session["messages"]
    current_messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.spinner("Consulting Research Database..."):
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAssistant: Provide a precise medical answer."
        response = llm.invoke(prompt)
        
        citations = list(set([build_apa_citation(d.metadata) for d in docs]))
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
            with st.expander("📎 Grounding Sources (APA)"):
                for cit in citations: st.caption(f"📍 {cit}")
        
        current_messages.append({
            "role": "assistant", 
            "content": response.content,
            "citations": citations
        })
