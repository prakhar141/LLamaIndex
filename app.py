import os
import logging
import streamlit as st
import google.generativeai as genai
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import nltk

# ============================== Environment Setup ==============================
os.makedirs("nltk_data", exist_ok=True)
os.environ["NLTK_DATA"] = os.path.join(os.getcwd(), "nltk_data")

nltk.download("punkt", download_dir=os.environ["NLTK_DATA"])
from nltk.tokenize import sent_tokenize

# ============================== HuggingFace Auth ==============================
login(token=st.secrets["HF_TOKEN"])

# ============================== Gemini LLM Setup ==============================
class GeminiLLM:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def complete(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

# ============================== Embedding Model ==============================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-base-en")

class HuggingFaceEmbedding:
    def __init__(self, normalize=True):
        self.model = load_embedder()
        self.normalize = normalize

    def get_embedding(self, text):
        return self.model.encode(text, normalize_embeddings=self.normalize)

# ============================== Streamlit UI ==============================
st.set_page_config(page_title="📚 PDF Chatbot", page_icon="📘", layout="centered")
st.title("📚 PDF Chatbot")
st.markdown("Upload PDFs or text files and ask questions from them using Gemini!")

uploaded_files = st.file_uploader("📂 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ============================== File Parsing ==============================
texts = []

def chunk_text(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

if uploaded_files:
    with st.status("📄 Extracting text..."):
        for file in uploaded_files:
            file_path = os.path.join(DATA_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

            if file.name.endswith(".pdf"):
                doc = fitz.open(file_path)
                for page in doc:
                    raw_text = page.get_text()
                    texts.extend(chunk_text(raw_text, max_words=80))
            elif file.name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                    texts.extend(chunk_text(raw_text, max_words=80))

    st.success("✅ Text extracted from uploaded files.")

# ============================== Model Init ==============================
llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
embedder = HuggingFaceEmbedding()

# ============================== Simple Vector Retrieval ==============================
if texts:
    embedded_texts = [(text, embedder.get_embedding(text)) for text in texts]

    user_input = st.text_input("💬 Ask a question from the documents:")
    if user_input:
        with st.spinner("🤖 Thinking..."):
            query_vec = embedder.get_embedding(user_input)

            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            similarities = [(text, cosine_similarity([vec], [query_vec])[0][0]) for text, vec in embedded_texts]
            top_match = max(similarities, key=lambda x: x[1])[0]

            prompt = f"""Answer the question using the context below.

Context: {top_match}

Question: {user_input}

Answer:"""
            try:
                answer = llm.complete(prompt)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("📌 Please upload PDF or TXT files to start.")
