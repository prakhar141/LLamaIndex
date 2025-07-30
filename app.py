import os
import fitz  # PyMuPDF
import streamlit as st
import requests
import tempfile
import subprocess
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== API Setup ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"

# ========== UI Setup ==========
st.set_page_config(page_title="📄 Quiliffy - Chat with PDFs", layout="wide")
st.title("📘 Welcome to Quiliffy")
st.markdown("""
Quiliffy lets you **chat with your PDFs**.  
Upload one or more PDF files, and ask anything related to their content! 💬
""")

# ========== File Upload ==========
uploaded_files = st.file_uploader("📄 Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# ========== Reset ==========
if st.button("🔁 Reset Session"):
    st.session_state.clear()
    st.rerun()

# ========== Utility: Check Poppler ==========
def check_poppler_installed():
    try:
        subprocess.run(["pdftoppm", "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

# ========== OCR for Scanned PDFs ==========
def extract_text_with_ocr(pdf_path):
    if not check_poppler_installed():
        st.error("❌ Poppler is not installed. OCR cannot run.")
        st.info("""
### How to install Poppler:
- 🐧 Ubuntu/Debian: `sudo apt install poppler-utils`
- 🍎 macOS: `brew install poppler`
- 🪟 Windows: [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
→ Add the `bin/` folder to your System PATH.
        """)
        return ""

    text = ""
    try:
        images = convert_from_path(pdf_path)
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"
    except Exception as e:
        st.error(f"🧨 OCR failed: {e}")
    return text

# ========== PDF Processor ==========
@st.cache_resource(show_spinner="📚 Indexing PDF(s)... Please wait.")
def build_vector_db(uploaded_files):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    for file in uploaded_files:
        st.markdown(f"🗂️ `{file.name}`")
        file.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Try direct text extraction
        with fitz.open(tmp_path) as doc:
            text = "\n".join([page.get_text() for page in doc])

        # If no text, fall back to OCR
        if not text.strip():
            st.warning(f"⚠️ `{file.name}` looks scanned. Running OCR…")
            text = extract_text_with_ocr(tmp_path)
        else:
            st.success(f"✅ Extracted text from `{file.name}` ({len(doc)} pages).")

        if not text.strip():
            st.error(f"❌ Couldn't extract text from `{file.name}`. Skipping.")
            continue

        # Split into chunks and store
        chunks = splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata={"source": file.name}) for chunk in chunks]
        all_docs.extend(docs)

    if not all_docs:
        st.error("🚫 No valid content found in uploaded PDFs.")
        return None

    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(all_docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=4)

# ========== LLM Query ==========
def ask_deepseek(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "PDF Chatbot"
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    payload = {"model": MODEL_NAME, "messages": messages}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {str(e)}"

# ========== Main Chat ==========
if uploaded_files:
    retriever = build_vector_db(uploaded_files)

    if not retriever:
        st.stop()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.chat_input("💬 Ask something about the document…")

    if query:
        with st.spinner("🤖 Thinking..."):
            try:
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                answer = ask_deepseek(context, query)
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
            st.session_state.chat.append({
                "question": query,
                "answer": answer,
                "sources": [doc.metadata["source"] for doc in docs]
            })

    for chat in reversed(st.session_state.chat):
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            for src in set(chat["sources"]):
                st.caption(f"📄 Source: `{src}`")

    with st.expander("🕘 View Chat History"):
        for i, chat in enumerate(st.session_state.chat):
            st.markdown(f"**Q{i+1}:** {chat['question']}")
            st.markdown(f"**A{i+1}:** {chat['answer']}")
            st.markdown("---")
else:
    st.info("📥 Please upload at least one PDF to begin chatting.")

# ========== Footer ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)
