import streamlit as st
import os

if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

import base64
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# ── الديزاين و كده ──────────────────────────
st.set_page_config(page_title="Virotropic AI", page_icon="🔬", layout="centered")

def get_logo_base64(path):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return None

logo_src = get_logo_base64("logo.png")
logo_html = f'<img src="{logo_src}" style="height:52px; vertical-align:middle; margin-right:12px;">' if logo_src else "🔬"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500&display=swap');

/* تنسيق الخلفية العامة */
.stApp {{ background-color: #FAF4E8; }}

/* الهيرو سيكشن */
.hero {{ 
    display: flex; align-items: center; justify-content: center; flex-direction: column; 
    padding: 2rem 0; border-bottom: 2px solid #722F37; margin-bottom: 2rem; text-align: center; 
}}
.hero h1 {{ font-family: 'Playfair Display', serif; font-size: 2.8rem; color: #722F37; margin: 0; }}
.hero p {{ font-family: 'Inter', sans-serif; color: #5E3A3E; font-weight: 400; }}

/* ستايل الشات مع إضافة الظلال (Shadows) */
[data-testid="stChatMessage"] {{
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border-radius: 16px;
    margin-bottom: 1rem;
    line-height: 1.6; /* تحسين المسافة بين الأسطر للقراءة الطبية */
}}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {{ 
    background-color: #722F37 !important; 
    color: #FAF4E8 !important; 
    padding: 15px;
    border-radius: 16px 16px 4px 16px !important;
}}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {{ 
    background-color: #FFFFFF !important; 
    border-left: 6px solid #722F37 !important; 
    color: #2D1B1E !important; 
    padding: 15px;
    border-radius: 4px 16px 16px 16px !important;
}}

/* تنسيق المراجع */
.citation-box {{
    background-color: #fff;
    border: 1px solid #e0e0e0;
    padding: 10px;
    border-radius: 8px;
    margin-top: 5px;
}}
</style>

<div class="hero">
    <div class="hero-top"> {logo_html} <h1>Virotropic</h1> </div>
    <p>Advanced Digital Repository & Medical AI Assistant</p>
</div>
""", unsafe_allow_html=True)

# ── (APA Logic) ──────────────────────────
def build_apa_citation(metadata, doc_content):
    author = metadata.get('author') or metadata.get('creator') or "ViroTropic Research Group"
    year = "2026" 
    title = metadata.get('title')
    
    if not title or title.lower() in ["none", "untitled"] or title.replace('.pdf','').isdigit():
        first_line = doc_content.split('\n')[0].strip()
        title = first_line[:80] + "..." if len(first_line) > 10 else f"Research_Doc_{os.path.basename(metadata.get('source', 'Ref'))}"
    
    return f"{author}. ({year}). {title}. ViroTropic Digital Repository."

# ──  RAG Setup ──────────────────────────────────────────
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "virotropic1"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME, 
    embedding=embeddings, 
    pinecone_api_key=os.environ["PINECONE_API_KEY"]
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# ── ويلكم يا عزيزي اتفضل اتفرج علي القماش──────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Welcome to **ViroTropic AI**. 🔬\n\nI am your specialized research assistant for Tropical Medicine, How can I assist your research today?"
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about Tropical Medicine, Malaria, or Research papers..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Consulting Virotropic Archives..."):
            try:
                search_results = vector_store.similarity_search(query, k=5)
                context = "\n\n---\n\n".join([doc.page_content for doc in search_results])

                prompt_template = f"""
                You are an expert medical research assistant specializing in Tropical Medicine.
                Provide a detailed answer based ONLY on the provided context.
                
                Context:
                {context}
                
                Question: {query}
                """
                
                response = llm.invoke(prompt_template)
                final_answer = response.content
                
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

                #  المراجع
                st.markdown("---")
                st.markdown("<small>📝 <b>Evidence-Based Citations (APA):</b></small>", unsafe_allow_html=True)
                
                seen_citations = set()
                for doc in search_results:
                    citation = build_apa_citation(doc.metadata, doc.page_content)
                    if citation not in seen_citations:
                        seen_citations.add(citation)
                        st.caption(f"📖 {citation}")

                with st.expander("Source Document Chunks"):
                    for i, doc in enumerate(search_results):
                        st.write(f"**Source {i+1}:** {doc.page_content[:400]}...")
                        st.divider()

            except Exception as e:
                st.error(f"An error occurred: {e}")
