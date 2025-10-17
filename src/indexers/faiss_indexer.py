from __future__ import annotations
from pathlib import Path
import faiss, numpy as np, pandas as pd
from loguru import logger
from typing import List
from ..schemas import Chunk

class FaissIndexer:
    def __init__(self, index_dir: Path, index_name: str, dim: int):
        self.index_path = index_dir / f"{index_name}.faiss"
        self.meta_path  = index_dir / f"{index_name}.parquet"
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.df = pd.DataFrame(columns=["chunk_id","doc_id","text","metadata"])
        if self.index_path.exists() and self.meta_path.exists():
            logger.info("Loading existing FAISS index + metadata")
            self.index = faiss.read_index(str(self.index_path))
            self.df = pd.read_parquet(self.meta_path)
    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        assert embeddings.shape[1] == self.dim
        existing = set(self.df["chunk_id"]) if not self.df.empty else set()
        rows, vecs = [], []
        for ch, vec in zip(chunks, embeddings):
            if ch.chunk_id in existing: continue
            rows.append({"chunk_id": ch.chunk_id, "doc_id": ch.doc_id, "text": ch.text, "metadata": ch.metadata})
            vecs.append(vec)
        if not rows: return
        self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
        self.index.add(np.vstack(vecs).astype("float32"))
    def persist(self):
        faiss.write_index(self.index, str(self.index_path))
        self.df.to_parquet(self.meta_path, index=False)
