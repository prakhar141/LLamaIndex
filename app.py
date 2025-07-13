import os
import streamlit as st
import fitz  # PyMuPDF
import tempfile
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List


# =================== Custom Gemini LLM Wrapper ===================
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"
    api_key: str = ""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "custom-gemini"


# =================== Helper: PDF Loader ===================
def load_pdf_chunks(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    texts = splitter.split_text(full_text)
    return [Document(page_content=t) for t in texts]


# =================== Streamlit UI ===================
st.set_page_config(page_title="📚 Snipurr ", page_icon="🧠")
st.title("📚 Talk to your PDF")

uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])
query = st.text_input("💬 Ask a question from your PDF:")

# =================== Main Logic ===================
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load & chunk
    with st.spinner("📄 Reading and chunking PDF..."):
        chunks = load_pdf_chunks(file_path)

    # Embedding & Index
    with st.spinner("🔎 Reading..."):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        vectordb = FAISS.from_documents(chunks, embeddings)
        retriever = vectordb.as_retriever(search_type="similarity", k=3)

    # Gemini LLM
    llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])

    # QA Chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        with st.spinner("🤖 Thinking it..."):
            answer = qa.run(query)
            st.success("🧠 Answer:")
            st.write(answer)
