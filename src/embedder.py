from __future__ import annotations
from typing import List
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt
from loguru import logger
from .config import settings

class BaseEmbedder:
    dim: int
    def embed(self, texts: List[str]) -> np.ndarray: ...

class STEmbedder(BaseEmbedder):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self):
        import openai
        self.client = openai.OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.model = settings.openai_deployment_name or "text-embedding-3-large"
        self.dim = 3072
    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def embed(self, texts: List[str]) -> np.ndarray:
        logger.debug(f"Embedding batch size={len(texts)}")
        resp = self.client.embeddings.create(input=texts, model=self.model)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

def make_embedder() -> BaseEmbedder:
    return OpenAIEmbedder() if settings.openai_api_key else STEmbedder()
