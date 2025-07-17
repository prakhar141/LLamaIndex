import streamlit as st
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, List, ClassVar
import torch
from pydantic import PrivateAttr


# ========== Custom LaMini-Flan LLM Wrapper with Token Safety ==========
class LaMiniFlanLLM(LLM):
    model_id: ClassVar[str] = "MBZUAI/LaMini-Flan-T5-783M"

    _tokenizer: any = PrivateAttr()
    _model: any = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Token-safe input
        inputs = self._tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=256)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "lamini-flan-t5"

# ========== Streamlit UI Setup ==========
st.set_page_config(page_title="🎓 Quillify", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        .big-title { font-size: 36px; font-weight: 800; margin-bottom: 10px; color: #3B82F6; }
        .subtitle { font-size: 16px; color: gray; margin-top: -10px; }
        .stTextInput > div > div > input { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-title'>🎓 Quillify</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a PDF and ask questions about it!</div>", unsafe_allow_html=True)

# ========== File Upload ==========
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

# ========== Process Uploaded PDF ==========
@st.cache_resource(show_spinner="📚 Reading PDF...")
def process_pdf(uploaded_file):
    if uploaded_file is None:
        return None

    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb.as_retriever(search_type="similarity", k=2)  # Reduced to avoid token overflow

# ========== Retrieval Chain ==========
if uploaded_file:
    retriever = process_pdf(uploaded_file)
    llm = LaMiniFlanLLM()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.chat_input("💬 Ask a question about your uploaded PDF")

    if query:
        with st.spinner("🤖 Thinking..."):
            answer = qa_chain.invoke(query)
            st.session_state.chat.append({"question": query, "answer": answer})

    # ========== Display Chat History ==========
    for entry in reversed(st.session_state.chat):
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

# ========== Footer ==========
st.markdown("""
    <hr style="margin-top: 40px; margin-bottom: 10px;">
    <div style='text-align: center; color: #aaa; font-size: 14px;'>
        🤖 Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani ·
    </div>
""", unsafe_allow_html=True)
