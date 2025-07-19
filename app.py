import os
import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import tempfile

# ======= OpenAI (ChatAnywhere) API Setup =======
client = OpenAI(
    api_key="YOUR_API_KEY",  # 🔁 Replace with your ChatAnywhere API key
    base_url="https://api.chatanywhere.tech/v1"
)

# ========== UI Setup ==========
st.set_page_config(page_title="🎓 Quillify", layout="wide")
st.markdown("""
    <style>
        .big-title { font-size: 36px; font-weight: 800; margin-bottom: 10px; color: #3B82F6; }
        .subtitle { font-size: 16px; color: gray; margin-top: -10px; }
        .stTextInput > div > div > input { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-title'>🎓 Quillify</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a PDF and ask questions about it</div>", unsafe_allow_html=True)

# ========== PDF Upload ==========
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

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
@st.cache_resource(show_spinner="Reading PDF...")
def build_vector_db_from_uploaded_pdf(file):
    docs = process_pdf(file)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb.as_retriever(search_type="similarity", k=4)

# ========== OpenAI Chat ==========
@st.cache_resource(show_spinner=" Loading LLM...")
def load_openai_client():
    return client

# ========== Chat System ==========
if uploaded_file:
    retriever = build_vector_db_from_uploaded_pdf(uploaded_file)
    openai_client = load_openai_client()

    def get_answer(query):
        context_docs = retriever.get_relevant_documents(query)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        system_prompt = f"You are an AI assistant. Use the following context to answer user questions.\n\nContext:\n{context_text}"

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.chat_input("💬 Ask me anything about this PDF...")

    if query:
        with st.spinner(" Thinking..."):
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
         Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani ·
        <br>📬 Feedback or queries? Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
    </div>
""", unsafe_allow_html=True)
