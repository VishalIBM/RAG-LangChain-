from __future__ import annotations
from typing import List, Optional
import os
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, VectorSearchAlgorithmConfiguration, VectorSearchAlgorithmKind,
    SemanticConfiguration, SemanticPrioritizedFields
)
from azure.search.documents import SearchClient
from ..schemas import Chunk
from ..config import settings

def _build_index_schema(index_name: str, vector_dim: int, semantic_config_name: Optional[str] = None) -> SearchIndex:
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchField(name="text", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="file_path", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="doc_title", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
        SimpleField(name="chunk_strategy", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dim,
            vector_search_configuration="vec-default",
        ),
    ]
    vector_search = VectorSearch(algorithm_configurations=[
        VectorSearchAlgorithmConfiguration(name="vec-default", kind=VectorSearchAlgorithmKind.HNSW)
    ])
    semantic_configuration = None
    if semantic_config_name:
        semantic_configuration = SemanticConfiguration(
            name=semantic_config_name,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=None, 
                content_fields=[{"name": "text"}], 
                keywords_fields=[]
            )
        )
    return SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_configurations=[semantic_configuration] if semantic_configuration else None)

class AzureSearchIndexer:
    def __init__(self, dim: int):
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") or ""
        api_key  = os.getenv("AZURE_SEARCH_API_KEY") or ""
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME") or settings.faiss_index_name
        sem_cfg = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG") or None
        override_dim = os.getenv("AZURE_SEARCH_VECTOR_DIM")

        if override_dim:
            try: dim = int(override_dim)
            except Exception: pass

        if not endpoint or not api_key or not index_name:
            raise ValueError("Set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME")

        self.index_name = index_name
        self.dim = dim
        self.credential = AzureKeyCredential(api_key)
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=self.credential)
        self.search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=self.credential)
        self._ensure_index(semantic_config_name=sem_cfg)

    def _ensure_index(self, semantic_config_name: Optional[str] = None):
        try:
            existing = self.index_client.get_index(self.index_name)
            vec = next((f for f in existing.fields if f.name == "vector"), None)
            if not vec or getattr(vec, "vector_search_dimensions", None) != self.dim:
                self.index_client.delete_index(self.index_name)
                self.index_client.create_index(_build_index_schema(self.index_name, self.dim, semantic_config_name))
                return
            logger.info(f"Azure index '{self.index_name}' present (dim={self.dim})")
        except Exception:
            logger.info(f"Creating Azure index '{self.index_name}' (dim={self.dim})")
            self.index_client.create_index(_build_index_schema(self.index_name, self.dim, semantic_config_name))

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _upload_batch(self, docs: list[dict]):
        actions = [{**d, "@search.action": "mergeOrUpload"} for d in docs]
        res = self.search_client.upload_documents(actions)
        failed = [r for r in res if not r.succeeded]
        if failed:
            raise RuntimeError(f"Azure Search batch failures: {len(failed)} (first={failed[0]})")

    def add(self, chunks: List[Chunk], embeddings) -> None:
        vectors = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        docs = []
        for ch, vec in zip(chunks, vectors):
            md = ch.metadata or {}
            docs.append({
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "doc_id": ch.doc_id,
                "file_name": md.get("file_name"),
                "file_path": md.get("file_path"),
                "doc_title": md.get("doc_title"),
                "page": int(md.get("page") or md.get("orig_page") or 0),
                "chunk_strategy": md.get("chunk_strategy"),
                "vector": [float(x) for x in vec],
            })
        B = 1000
        for i in range(0, len(docs), B):
            self._upload_batch(docs[i:i+B])

    def persist(self): pass
