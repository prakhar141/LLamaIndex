import os
import time
import logging
import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import TooManyRequests

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.settings import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings


# ====== Gemini LLM Setup ======
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


# ====== Logging & Retry Setup ======
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


# ====== Streamlit UI ======
st.title("📚 Gemini PDF Chatbot")
st.markdown("Upload PDFs or Text files and ask questions!")

# File upload
uploaded_files = st.file_uploader("Upload your files", type=["pdf", "txt"], accept_multiple_files=True)

# Persist uploaded files
if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.success("✅ Files saved and ready for indexing.")

# Load & build index if files are present
if os.path.exists("data") and any(os.scandir("data")):
    # Setup Gemini LLM
    llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
    Settings.llm = llm

    # Setup embeddings
    lc_embed = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    Settings.embed_model = LangchainEmbedding(lc_embed)

    # Optional: chunk size
    Settings.chunk_size = 800
    Settings.chunk_overlap = 20

    # Load documents and create index
    with st.spinner("🔍 Indexing documents..."):
        try:
            documents = SimpleDirectoryReader("data").load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist()
            query_engine = index.as_query_engine()
            st.success("✅ Index created successfully.")
        except Exception as e:
            st.error(f"❌ Error during indexing: {e}")
            st.stop()

    # Ask question
    user_question = st.text_input("💬 Ask a question from the documents:")
    if user_question:
        with st.spinner("🤖 Thinking..."):
            answer = query_index_with_retry(query_engine, user_question)
            st.markdown(f"**Answer:** {answer}")
