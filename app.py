import os
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import tempfile

# ======= Gemini API Key Setup =======
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or "YOUR_GEMINI_API_KEY")

# ========== UI Setup ==========
st.set_page_config(page_title="🎓 Quillify", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        .big-title { font-size: 36px; font-weight: 800; margin-bottom: 10px; color: #3B82F6; }
        .subtitle { font-size: 16px; color: gray; margin-top: -10px; }
        .stTextInput > div > div > input { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-title'>🎓 Quillify</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a BITS PDF and ask questions about it</div>", unsafe_allow_html=True)

# ========== PDF Upload ==========
uploaded_file = st.file_uploader("📄 Upload your BITS PDF", type=["pdf"])

# ========== PDF Processing ==========
def process_pdf(file):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    with fitz.open(tmp_path) as doc:
        full_text = "\n".join([page.get_text() for page in doc])
    chunks = splitter.split_text(full_text)
    return [Document(page_content=chunk) for chunk in chunks]

# ========== Build Vector DB ==========
@st.cache_resource(show_spinner="📚 Indexing PDF...")
def build_vector_db_from_uploaded_pdf(file):
    docs = process_pdf(file)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb.as_retriever(search_type="similarity", k=4)

# ========== Gemini Chat ==========
@st.cache_resource(show_spinner="🤖 Loading Gemini...")
def load_gemini_model():
    return genai.GenerativeModel("gemini-1.5-flash")

# ========== Chat System ==========
if uploaded_file:
    retriever = build_vector_db_from_uploaded_pdf(uploaded_file)
    gemini = load_gemini_model()

    def get_answer(query):
        context_docs = retriever.get_relevant_documents(query)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = f"""Answer the following question based on the given context.\n\nContext:\n{context_text}\n\nQuestion: {query}"""
        response = gemini.generate_content(prompt)
        return response.text

    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.chat_input("💬 Ask me anything about this PDF...")

    if query:
        with st.spinner("🤖 Thinking..."):
            try:
                answer = get_answer(query)
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
            st.session_state.chat.append({"question": query, "answer": answer})

    for entry in reversed(st.session_state.chat):
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
else:
    st.warning("📥 Please upload a PDF to begin.")

# ========== Footer ==========
st.markdown("""
    <hr style="margin-top: 40px; margin-bottom: 10px;">
    <div style='text-align: center; color: #aaa; font-size: 14px;'>
        🤖 Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani ·
    </div>
""", unsafe_allow_html=True)
