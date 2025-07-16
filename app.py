import os
import fitz  # PyMuPDF
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# ========= Load PDFs =========
def load_all_pdfs():
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    for file in os.listdir():
        if file.endswith(".pdf"):
            with fitz.open(file) as doc:
                full_text = "\n".join([page.get_text() for page in doc])
            chunks = splitter.split_text(full_text)
            docs.extend([Document(page_content=chunk, metadata={"source": file}) for chunk in chunks])
    return docs

# ========= UI Config =========
st.set_page_config(page_title="🎓 Quillify", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        .big-title { font-size: 36px; font-weight: 800; margin-bottom: 10px; color: #3B82F6; }
        .subtitle { font-size: 16px; color: gray; margin-top: -10px; }
        .stTextInput > div > div > input { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-title'>🎓 Quillify</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask anything about BITS – syllabus, events, academics, policies, and more</div>", unsafe_allow_html=True)

# ========= Load Vector DB =========
@st.cache_resource(show_spinner="📚 Reading PDFs...")
def setup_vector_db():
    documents = load_all_pdfs()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb.as_retriever(search_type="similarity", k=4)

retriever = setup_vector_db()

# ========= Load LaMini-Flan-T5-783M =========
@st.cache_resource(show_spinner="🤖 Booting up LLM...")
def load_llm():
    model_id = "MBZUAI/LaMini-Flan-T5-783M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# ========= Retrieval Chain =========
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ========= Chat State =========
if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("💬 I know more about BITS than your CGPA does.")

if query:
    with st.spinner("🤖 Thinking..."):
        answer = qa_chain.run(query)
        st.session_state.history.append((query, answer))

        # User's question
        st.chat_message("user").markdown(query)

        # Assistant's answer
        with st.chat_message("assistant"):
            st.markdown(answer)

            # Feedback + Download
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("👍 Helpful"):
                    st.toast("Thanks for your feedback!")
            with col2:
                if st.button("👎 Not Helpful"):
                    st.toast("We'll work on it!")
            with col3:
                st.download_button("📥 Download Answer", answer, file_name="response.txt")

# ========= Chat History =========
if st.session_state.history:
    with st.expander("🕓 Chat History"):
        for q, a in reversed(st.session_state.history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

# ========= Footer =========
st.markdown("""
    <hr style="margin-top: 30px; margin-bottom: 10px;">
    <div style='text-align: center; color: gray; font-size: 14px;'>
        🤖 Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani
    </div>
""", unsafe_allow_html=True)
