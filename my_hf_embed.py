from llama_index.core.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer

class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, device: str = "cpu", embed_batch_size: int = 8, normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.device = device
        self.embed_batch_size = embed_batch_size
        self.normalize = normalize

    def _get_text_embedding(self, text: str):
        return self.model.encode(text, normalize_embeddings=self.normalize)

    def _get_text_embeddings(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=self.normalize)
