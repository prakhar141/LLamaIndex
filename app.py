import os
import time
import logging
import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import TooManyRequests

from llama_index.readers.file import SimpleDirectoryReader
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage, ServiceContext
from llama_index.llms.base import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.settings import Settings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings


# ============================== Gemini LLM Setup ==============================
class GeminiLLM(CustomLLM):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        super().__init__()
        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(model)

    @property
    def context_window(self) -> int:
        return 8192

    @property
    def num_output(self) -> int:
        return 512

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self._model_name,
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self._model.generate_content(prompt)
        return CompletionResponse(text=response.text)

    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming not supported.")


# ============================== Logging & Retry ==============================
logging.basicConfig(level=logging.INFO)

def query_index_with_retry(engine, prompt, max_retries=5, backoff=2):
    attempt = 0
    while attempt < max_retries:
        try:
            response = engine.query(prompt)
            return response.response
        except TooManyRequests:
            logging.warning(f"Rate limit hit. Retry {attempt+1}/{max_retries}")
            time.sleep(backoff * (attempt + 1))
            attempt += 1
        except Exception as e:
            logging.error(f"Query error: {e}")
            break
    return "⚠️ Failed after retries."


# ============================== Streamlit UI ==============================
st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="📚", layout="centered")
st.title("📚 Gemini PDF Chatbot")
st.markdown("Upload your PDFs or text files and ask questions from their content!")

# Upload area
uploaded_files = st.file_uploader("📂 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

DATA_DIR = "data"
INDEX_DIR = "storage"

# Save files if uploaded
if uploaded_files:
    os.makedirs(DATA_DIR, exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.success("✅ Files uploaded and saved.")

# Setup Gemini + Embeddings
llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
embed_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

Settings.llm = llm
Settings.embed_model = LangchainEmbedding(embed_model)
Settings.chunk_size = 800
Settings.chunk_overlap = 20

# Build or load index
if os.path.exists(DATA_DIR) and any(os.scandir(DATA_DIR)):
    if os.path.exists(INDEX_DIR):
        try:
            st.info("📦 Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            index = load_index_from_storage(storage_context)
            query_engine = index.as_query_engine()
            st.success("✅ Index loaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to load existing index: {e}")
            st.stop()
    else:
        with st.spinner("🔍 Indexing uploaded files..."):
            try:
                documents = SimpleDirectoryReader(DATA_DIR).load_data()
                index = VectorStoreIndex.from_documents(documents)
                index.storage_context.persist(persist_dir=INDEX_DIR)
                query_engine = index.as_query_engine()
                st.success("✅ New index created successfully.")
            except Exception as e:
                st.error(f"❌ Error while indexing: {e}")
                st.stop()

    # Chat interface
    user_input = st.text_input("💬 Ask your question from the documents:")
    if user_input:
        with st.spinner("🤖 Thinking..."):
            answer = query_index_with_retry(query_engine, user_input)
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("📎 Please upload files to start chatting.")
