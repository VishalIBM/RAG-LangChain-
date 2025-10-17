from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class AzureSearchRetriever:
    def __init__(self, vector_dim: int):
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") or ""
        api_key  = os.getenv("AZURE_SEARCH_API_KEY") or ""
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME") or ""
        if not endpoint or not api_key or not index_name:
            raise ValueError("Azure Search config missing")
        self.client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
        self.use_hybrid = (os.getenv("AZURE_SEARCH_USE_HYBRID", "true").lower() == "true")
        self.vector_field = "vector"
        self.vector_dim = vector_dim

    def retrieve(self, query_embedding: list[float], k: int = 10,
                 search_text: Optional[str] = None, filter_expr: Optional[str] = None,
                 select_fields: Optional[List[str]] = None) -> List[RetrievalResult]:
        if not select_fields:
            select_fields = ["chunk_id","text","doc_id","file_name","file_path","doc_title","page","chunk_strategy"]
        if self.use_hybrid and (search_text or "").strip():
            results = self.client.search(
                search_text=search_text,
                vector={"value": query_embedding, "fields": self.vector_field, "k": k},
                filter=filter_expr, select=select_fields, top=k
            )
        else:
            results = self.client.search(
                search_text=None,
                vector={"value": query_embedding, "fields": self.vector_field, "k": k},
                filter=filter_expr, select=select_fields, top=k
            )
        out: List[RetrievalResult] = []
        for r in results:
            meta = {k: r.get(k) for k in ["doc_id","file_name","file_path","doc_title","page","chunk_strategy"]}
            out.append(RetrievalResult(
                chunk_id=r["chunk_id"], score=float(getattr(r, "@search.score", 0.0)),
                text=r.get("text",""), metadata=meta
            ))
        return out
