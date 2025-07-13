import os
import time
import logging
import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import TooManyRequests
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import nltk

# Fix: Download punkt and ensure everything is cached
nltk.download("punkt")
nltk.download("punkt_tab")  # ✅ explicitly add this line
from nltk.tokenize import sent_tokenize

# ============================== Environment Setup ==============================
os.makedirs("nltk_data", exist_ok=True)
os.environ["NLTK_DATA"] = os.path.join(os.getcwd(), "nltk_data")

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
class HuggingFaceEmbedding:
    def __init__(self, model_name="BAAI/bge-base-en", device="cpu", normalize=True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def get_embedding(self, text):
        return self.model.encode(text, normalize_embeddings=self.normalize)

# ============================== Streamlit UI ==============================
st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="centered")
st.title("📚 PDF Chatbot")
st.markdown("Upload your PDFs or text files and ask questions from their content!")

uploaded_files = st.file_uploader("📂 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Save files
texts = []
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".pdf"):
            doc = fitz.open(file_path)
            for page in doc:
                texts.extend(sent_tokenize(page.get_text()))
        elif file.name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.extend(sent_tokenize(f.read()))

    st.success("✅ Files uploaded and text extracted.")

# Initialize models
llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
embedder = HuggingFaceEmbedding()

# Simple Retrieval (no llama-index)
if texts:
    # Compute embeddings once
    embedded_texts = [(text, embedder.get_embedding(text)) for text in texts]

    user_input = st.text_input("💬 Ask a question from the documents:")
    if user_input:
        with st.spinner("🤖 Thinking..."):
            query_vec = embedder.get_embedding(user_input)

            # Compute cosine similarity manually
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            similarities = [(text, cosine_similarity([vec], [query_vec])[0][0]) for text, vec in embedded_texts]
            top_match = max(similarities, key=lambda x: x[1])[0]

            prompt = f"Answer this question based on the following context:\n\nContext: {top_match}\n\nQuestion: {user_input}\n\nAnswer:"
            answer = llm.complete(prompt)
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("📌 Please upload files to start.")
