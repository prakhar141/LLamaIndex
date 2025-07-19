import os
import fitz  # PyMuPDF
import streamlit as st
import requests
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== API Setup ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat"  # ✅ Free model

# ========== UI ==========
st.set_page_config(page_title="📄 Quiliffy", layout="wide")
st.title("📘 Chat with your PDF")
st.markdown("Upload a PDF and chat with it ")

# ========== PDF Upload ==========
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

# ========== PDF Processing ==========
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    with fitz.open(tmp_path) as doc:
        text = "\n".join([page.get_text() for page in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

@st.cache_resource(show_spinner="📚 Processing PDF...")
def build_vector_db(file):
    docs = process_pdf(file)
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=4)

# ========== Chat Function ==========
def ask_deepseek(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",  # update with your site if hosted
        "X-Title": "PDF Chatbot via DeepSeek"
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    payload = {"model": MODEL_NAME, "messages": messages}
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# ========== Chat UI ==========
if uploaded_file:
    retriever = build_vector_db(uploaded_file)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.chat_input("💬 Ask a question from the PDF")

    if query:
        with st.spinner("🤖 Thinking..."):
            try:
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                answer = ask_deepseek(context, query)
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
            st.session_state.chat.append({"question": query, "answer": answer})

    for chat in reversed(st.session_state.chat):
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
else:
    st.info("📥 Please upload a PDF to begin chatting.")

# ========== Footer ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)
