from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class RerankItem:
    text: str
    meta: dict

class LocalCrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    def rerank(self, query: str, items: List[RerankItem], top_k: int) -> List[Tuple[int, float]]:
        if not items: return []
        pairs = [(query, it.text) for it in items]
        scores = self.model.predict(pairs)
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]
