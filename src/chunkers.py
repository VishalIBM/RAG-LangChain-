from __future__ import annotations
from typing import List
import re, math
from .schemas import Document, Chunk
from .utils import stable_hash, count_tokens

class BaseChunker:
    def chunk(self, doc: Document) -> List[Chunk]: raise NotImplementedError
    def _make_chunk(self, doc: Document, text: str, idx: int) -> Chunk:
        md = dict(doc.metadata); md.update({
            "chunk_index": idx,
            "orig_doc_id": doc.doc_id,
            "orig_page": doc.page,
            "chunk_strategy": self.__class__.__name__,  # used by Azure filter
        })
        cid = stable_hash(f"{doc.doc_id}::{idx}::{len(text)}")
        return Chunk(cid, doc.doc_id, text, md)

class CharacterChunker(BaseChunker):
    def __init__(self, target_tokens=400, overlap_tokens=40, token_model="cl100k_base"):
        self.target, self.overlap, self.token_model = target_tokens, overlap_tokens, token_model
    def chunk(self, doc: Document) -> List[Chunk]:
        words = doc.text.split(); chunks, start, idx = [], 0, 0
        while start < len(words):
            end, best_end, step = min(len(words), start+512), start, 512
            while True:
                cand = " ".join(words[start:end]); toks = count_tokens(cand, self.token_model)
                if toks >= self.target or end == len(words): break
                best_end, end, step = end, min(len(words), end+step), max(32, step//2)
            if best_end > start: end = best_end
            text = " ".join(words[start:end]).strip()
            if text: chunks.append(self._make_chunk(doc, text, idx)); idx += 1
            tok = max(1, count_tokens(text, self.token_model))
            overlap_words = max(0, math.floor(self.overlap * (end-start) / tok))
            start = max(end - overlap_words, start+1)
        return chunks

class RecursiveChunker(BaseChunker):
    def __init__(self, max_tokens=500, token_model="cl100k_base"):
        self.max_tokens, self.token_model = max_tokens, token_model
    def _split_structural(self, text: str) -> List[str]:
        blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
        out: List[str] = []
        for b in blocks:
            if count_tokens(b, self.token_model) <= self.max_tokens: out.append(b)
            else: out.extend([p for p in re.split(r"(?<=[\.!?])\s+", b) if p.strip()])
        return out
    def chunk(self, doc: Document) -> List[Chunk]:
        units = self._split_structural(doc.text); chunks, buf, buf_tok, idx = [], [], 0, 0
        for u in units:
            t = count_tokens(u, self.token_model)
            if t > self.max_tokens:
                if buf: chunks.append(self._make_chunk(doc, "\n\n".join(buf).strip(), idx)); idx += 1; buf, buf_tok = [], 0
                words, lo = u.split(), 0
                while lo < len(words):
                    hi = lo + 200; piece = " ".join(words[lo:hi])
                    while count_tokens(piece, self.token_model) > self.max_tokens and hi - lo > 20:
                        hi -= 20; piece = " ".join(words[lo:hi])
                    chunks.append(self._make_chunk(doc, piece, idx)); idx += 1; lo = hi
                continue
            if buf_tok + t <= self.max_tokens: buf.append(u); buf_tok += t
            else:
                chunks.append(self._make_chunk(doc, "\n\n".join(buf).strip(), idx)); idx += 1
                buf, buf_tok = [u], t
        if buf: chunks.append(self._make_chunk(doc, "\n\n".join(buf).strip(), idx))
        return chunks

class AdaptiveChunker(BaseChunker):
    def __init__(self, max_tokens=550, min_tokens=120, token_model="cl100k_base", coherence_threshold=0.55):
        from sentence_transformers import SentenceTransformer
        self.max_tokens, self.min_tokens, self.token_model, self.ct = max_tokens, min_tokens, token_model, coherence_threshold
        self.enc = SentenceTransformer("all-MiniLM-L6-v2")
    def _cos(self, a, b):
        import numpy as np
        a = a/(np.linalg.norm(a)+1e-12); b = b/(np.linalg.norm(b)+1e-12)
        return float((a*b).sum())
    def _units(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        units: List[str] = []
        for p in paras:
            if count_tokens(p, self.token_model) <= self.max_tokens: units.append(p)
            else: units.extend([s for s in re.split(r"(?<=[\.!?])\s+", p) if s.strip()])
        return units
    def chunk(self, doc: Document) -> List[Chunk]:
        units = self._units(doc.text); 
        if not units: return []
        embs = self.enc.encode(units, normalize_embeddings=True)
        chunks: List[Chunk] = []; i, idx = 0, 0
        while i < len(units):
            cur, cur_tok = units[i], count_tokens(units[i], self.token_model)
            if cur_tok >= self.max_tokens: chunks.append(self._make_chunk(doc, cur, idx)); idx += 1; i += 1; continue
            j, buf, buf_tok, last = i+1, [cur], cur_tok, embs[i]
            while j < len(units):
                cand, cand_tok = units[j], count_tokens(units[j], self.token_model)
                if buf_tok + cand_tok > self.max_tokens: break
                if self._cos(last, embs[j]) < self.ct and buf_tok >= self.min_tokens: break
                buf.append(cand); buf_tok += cand_tok; last = embs[j]; j += 1
            chunks.append(self._make_chunk(doc, " ".join(buf).strip(), idx)); idx += 1
            i = max(j, i+1)
        return chunks
