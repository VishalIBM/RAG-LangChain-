from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os, faiss, pandas as pd, numpy as np
from ..embedder import make_embedder
from ..config import settings
try:
    from ..retriever_azure import AzureSearchRetriever as _AzureRetriever
    _HAS_AZURE = True
except Exception:
    _HAS_AZURE = False

@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class FaissRetriever:
    def __init__(self, index_name: str):
        self.index = faiss.read_index(str(settings.index_dir / f"{index_name}.faiss"))
        self.df = pd.read_parquet(settings.index_dir / f"{index_name}.parquet")
        self.embedder = make_embedder()
    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        qv = self.embedder.embed([query]).astype("float32")
        D, I = self.index.search(qv, k)
        out: List[RetrievalResult] = []
        for idx, score in zip(I[0], D[0]):
            if int(idx) == -1: continue
            r = self.df.iloc[int(idx)]
            out.append(RetrievalResult(r["chunk_id"], float(score), r["text"], dict(r["metadata"])))
        return out

class Retriever:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.backend = os.getenv("INDEX_BACKEND","faiss").lower()
        self.embedder = make_embedder()
        if self.backend == "azure":
            if not _HAS_AZURE: raise RuntimeError("Azure retriever not available")
            self.azure = _AzureRetriever(vector_dim=self.embedder.dim)
            self.faiss = None
            self.filter_by_strategy = (os.getenv("AZURE_SEARCH_FILTER_BY_STRATEGY","true").lower()=="true")
        else:
            self.faiss = FaissRetriever(f"{settings.faiss_index_name}_{strategy}")
            self.azure = None
            self.filter_by_strategy = False
    def retrieve(self, question: str, k: int = 10) -> List[RetrievalResult]:
        if self.backend == "azure":
            qv = self.embedder.embed([question]).tolist()[0]
            filt = None
            if self.filter_by_strategy:
                name_map = {"character":"CharacterChunker","recursive":"RecursiveChunker","adaptive":"AdaptiveChunker"}
                filt = f"chunk_strategy eq '{name_map.get(self.strategy,self.strategy)}'"
            res = self.azure.retrieve(query_embedding=qv, k=k, search_text=question, filter_expr=filt)
            return [RetrievalResult(r.chunk_id, r.score, r.text, r.metadata) for r in res]
        return self.faiss.retrieve(question, k=k)
