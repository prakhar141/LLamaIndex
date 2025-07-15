import os
import tempfile
import urllib.parse
import fitz
import streamlit as st
import google.generativeai as genai
import wikipedia
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

# ========== Custom Gemini LLM Wrapper ==========
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"
    api_key: str = ""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        return model.generate_content(prompt).text

    @property
    def _llm_type(self) -> str:
        return "custom-gemini"

# ========== PDF Text Chunking ==========
def load_pdf_chunks(file_path, source_name):
    doc = fitz.open(file_path)
    text = "".join(page.get_text() for page in doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    return [Document(page_content=chunk, metadata={"source": source_name})
            for chunk in splitter.split_text(text)]

# ========== Streamlit Layout ==========
st.set_page_config(page_title="📚 Snipurr", page_icon="🧠")
st.title("📚 Talk to your PDF")
st.markdown("Upload PDFs and choose modes: QA, Summary, Keywords, Auto Q&A")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader("📂 Upload PDF files", type=["pdf"], accept_multiple_files=True)
query = st.text_input("💬 Ask something or leave blank for Q&A generation:")
mode = st.selectbox("🧭 Choose Mode", ["QA", "Summarize", "Keywords", "Generate Q&A"])
external = st.checkbox("🌐 Fetch Wikipedia summary & YouTube link")

# ========== Core Logic ==========
if uploaded_files:
    with st.spinner("📄 Extracting text from PDFs..."):
        all_chunks = []
        for file in uploaded_files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(file.read())
            all_chunks += load_pdf_chunks(tmp.name, file.name)

    with st.spinner("🔎 Building vector store..."):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en",
                                           model_kwargs={"device": "cpu"})
        vectordb = FAISS.from_documents(all_chunks, embeddings)
        retriever = vectordb.as_retriever(search_type="similarity", k=3)

    llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query or mode == "Generate Q&A":
        with st.spinner("🤖 Generating response..."):
            if mode == "QA":
                answer = qa.run(query)

            elif mode == "Summarize":
                docs = retriever.get_relevant_documents(query)
                context = "\n".join(doc.page_content for doc in docs)
                answer = llm._call(f"Summarize this:\n\n{context}")

            elif mode == "Keywords":
                docs = retriever.get_relevant_documents(query)
                context = "\n".join(doc.page_content for doc in docs)
                answer = llm._call(f"Extract keywords from this:\n\n{context}")

            elif mode == "Generate Q&A":
                context = "\n".join(doc.page_content for doc in all_chunks[:5])
                answer = llm._call(f"Generate 5 question-answer pairs from this:\n\n{context}")

            # Append answer
            st.session_state.history.append((f"{mode} → {query}", answer))
            st.success("🧠 Answer:")
            st.markdown(answer)

            # External enrichment
            if external and query:
                encoded = urllib.parse.quote_plus(query)
                st.markdown("### 🌐 Related Resources")
                # Wikipedia summary
                try:
                    summary = wikipedia.summary(query, sentences=2)
                    st.markdown(f"**Wikipedia Summary:** {summary}")
                except:
                    st.markdown("*No Wikipedia summary found.*")
                # YouTube link
                st.markdown(f"- 📺 [YouTube search](https://www.youtube.com/results?search_query={encoded})")

            # PDF contexts
            if mode in ["QA", "Summarize", "Keywords"]:
                with st.expander("📚 Supporting PDF Context"):
                    for doc in retriever.get_relevant_documents(query):
                        st.markdown(f"**Source:** {doc.metadata.get('source')}")
                        st.code(doc.page_content[:400])

            # Feedback + Download
            col1, col2 = st.columns(2)
            if col1.button("👍 Helpful"):
                st.toast("Thanks!")
            if col2.button("👎 Not Helpful"):
                st.toast("Will improve soon!")
            st.download_button("📥 Download Answer", answer, file_name="response.txt")

    with st.expander("🕓 Chat History"):
        for q_, a_ in reversed(st.session_state.history):
            st.markdown(f"**Q:** {q_}")
            st.markdown(f"**A:** {a_}\n---")
else:
    st.info("📌 Upload a PDF to get started.")

# ========== Footer ==========
st.markdown("""
<hr style="margin-top: 20px; margin-bottom: 10px;">
<div style='text-align: center; color: gray; font-size: 14px;'>
Developed with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani
</div>
""", unsafe_allow_html=True)
