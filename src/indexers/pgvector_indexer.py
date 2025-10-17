import json, numpy as np, psycopg2
from psycopg2.extras import execute_batch
from ..schemas import Chunk
from ..config import settings

class PgVectorIndexer:
    def __init__(self, dim: int):
        conn_str = settings.pgvectoR_conn_str or settings.PGVECTOR_CONN_STR
        if not conn_str: raise ValueError("PGVECTOR_CONN_STR not set")
        self.conn = psycopg2.connect(conn_str)
        self.table = settings.PGVECTOR_TABLE or "pdf_chunks"
        self.dim = dim
        with self.conn, self.conn.cursor() as cur:
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table}(
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                metadata JSONB,
                embedding VECTOR({self.dim})
            );
            """)
    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        rows = [(c.chunk_id, c.doc_id, c.text, json.dumps(c.metadata), list(map(float, embeddings[i])))
                for i, c in enumerate(chunks)]
        with self.conn, self.conn.cursor() as cur:
            execute_batch(cur, f"""
              INSERT INTO {self.table}(chunk_id,doc_id,text,metadata,embedding)
              VALUES (%s,%s,%s,%s,%s)
              ON CONFLICT (chunk_id) DO NOTHING
            """, rows)
    def persist(self): self.conn.commit()
