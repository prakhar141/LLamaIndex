import os
import tempfile
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

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

# =================== PDF Loader ===================
def load_pdf_chunks(file_path, source_name):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    texts = splitter.split_text(full_text)
    return [Document(page_content=t, metadata={"source": source_name}) for t in texts]

# =================== Streamlit UI ===================
st.set_page_config(page_title="📚 Snipurr", page_icon="🧠")
st.title("📚 Talk to your PDF")
st.markdown("Upload PDFs and chat, summarize, extract keywords, or download answers!")

# Chat history init
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader("📂 Upload PDF files", type=["pdf"], accept_multiple_files=True)
query = st.text_input("💬 Ask something or use a mode:")
mode = st.selectbox("Choose Mode", ["QA", "Summarize", "Keywords"])

# =================== Main Logic ===================
if uploaded_files:
    with st.spinner("📄 Reading and chunking PDFs..."):
        all_chunks = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                file_path = tmp_file.name
            chunks = load_pdf_chunks(file_path, file.name)
            all_chunks.extend(chunks)

    with st.spinner("🔎 Embedding..."):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        vectordb = FAISS.from_documents(all_chunks, embeddings)
        retriever = vectordb.as_retriever(search_type="similarity", k=3)

    llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        with st.spinner("🤖 Thinking..."):
            if mode == "QA":
                answer = qa.run(query)
            elif mode == "Summarize":
                docs = retriever.get_relevant_documents(query)
                context = "\n".join([doc.page_content for doc in docs])
                prompt = f"Summarize this content:\n\n{context}"
                answer = llm._call(prompt)
            elif mode == "Keywords":
                docs = retriever.get_relevant_documents(query)
                context = "\n".join([doc.page_content for doc in docs])
                prompt = f"Extract important keywords from this content:\n\n{context}"
                answer = llm._call(prompt)

            st.session_state.history.append((query, answer))
            st.success("🧠 Answer:")
            st.markdown(answer)

            # Supporting context
            with st.expander("📚 Supporting Contexts"):
                for doc in retriever.get_relevant_documents(query):
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    st.code(doc.page_content[:400])

            # Feedback
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful"):
                    st.toast("Thanks for your feedback!")
            with col2:
                if st.button("👎 Not Helpful"):
                    st.toast("We'll work on it!")

            # Download button
            st.download_button("📥 Download Answer", answer, file_name="response.txt")

    # History section
    with st.expander("🕓 Chat History"):
        for q, a in reversed(st.session_state.history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
else:
    st.info("📌 Upload a PDF to get started.")
