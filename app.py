import os
import fitz  # PyMuPDF
import streamlit as st
import requests
import tempfile
import pytesseract
from PIL import Image
from io import BytesIO
import time
import base64

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== Environment Setup ==========
# Set Tesseract path for Streamlit deployment
if os.name == "posix" and not os.path.exists("/usr/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/usr/share/tesseract-ocr/4.00/tessdata"

# ========== API Setup ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "YOUR_API_KEY")
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

# ========== Reset Button ==========
if st.button("🔁 Reset Session"):
    st.session_state.clear()
    st.rerun()

# ========== Enhanced OCR with Progress ==========
def extract_text_with_ocr(pdf_path):
    """Extract text from scanned PDFs using OCR with progress tracking"""
    full_text = ""
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                image = Image.open(BytesIO(img_bytes))
                
                # Preprocess image for better OCR
                image = image.convert('L')  # Convert to grayscale
                
                text = pytesseract.image_to_string(image)
                full_text += f"{text}\n\n"
                
                # Update progress
                progress = int((page_num + 1) / total_pages * 100)
                progress_bar.progress(progress)
                status_text.text(f"📖 Processing page {page_num+1}/{total_pages}...")
                time.sleep(0.01)  # Allow UI update
            
            progress_bar.empty()
            status_text.empty()
            
    except Exception as e:
        st.error(f"❌ OCR Error: {str(e)}")
    return full_text

# ========== PDF Processing ==========
@st.cache_resource(show_spinner="📚 Indexing PDF(s)... Please wait.")
def build_vector_db(uploaded_files):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    for file in uploaded_files:
        with st.expander(f"🗂️ Processing: `{file.name}`", expanded=False):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name

            try:
                # First attempt: Extract normal text
                with fitz.open(tmp_path) as doc:
                    text = "\n".join([page.get_text() for page in doc])
                
                # Check if text extraction failed (scanned PDF)
                if len(text.strip()) < 50:  # Threshold for considering empty
                    st.warning("⚠️ Low text count detected - using OCR for scanned PDF")
                    text = extract_text_with_ocr(tmp_path)
                    
                    if not text.strip():
                        st.error("❌ OCR failed to extract text")
                        continue
                    else:
                        st.success(f"✅ Extracted {len(text.split())} words via OCR")
                else:
                    st.success(f"✅ Extracted {len(text.split())} words")
                
                chunks = splitter.split_text(text)
                docs = [Document(page_content=chunk, metadata={"source": file.name}) for chunk in chunks]
                all_docs.extend(docs)
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                continue
            finally:
                os.unlink(tmp_path)

    if not all_docs:
        st.error("🚫 No extractable content found in uploaded PDFs")
        return None

    try:
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        vectordb = FAISS.from_documents(all_docs, embedder)
        return vectordb.as_retriever(search_type="similarity", k=4)
    except Exception as e:
        st.error(f"❌ Vector DB creation failed: {str(e)}")
        return None

# ========== Chat Function ==========
def ask_deepseek(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "PDF Chatbot"
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful PDF analysis assistant. Use the provided context to answer questions. "
                "Be concise but thorough. If the answer isn't in the context, say so."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        }
    ]
    payload = {"model": MODEL_NAME, "messages": messages}
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {str(e)}"

# ========== Main Chat Flow ==========
if uploaded_files:
    retriever = build_vector_db(uploaded_files)

    if not retriever:
        st.stop()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.chat_input("💬 Ask something about the document…")

    if query:
        with st.spinner("🔍 Searching documents..."):
            try:
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs])
                
                with st.spinner("🤖 Generating response..."):
                    answer = ask_deepseek(context, query)
                
                st.session_state.chat.append({
                    "question": query,
                    "answer": answer,
                    "sources": list(set(doc.metadata["source"] for doc in docs))
                })
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")

    for chat in st.session_state.chat:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            with st.expander("📄 Source Documents"):
                for src in chat["sources"]:
                    st.caption(f"`{src}`")

    with st.expander("🕘 View Full Chat History"):
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
