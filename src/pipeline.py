import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Literal, List
from loguru import logger
from src.config import settings
from src.logging_setup import setup_logging
from src.loader import load_pdfs
from src.chunkers import CharacterChunker, RecursiveChunker, AdaptiveChunker
from src.embedder import make_embedder
from src.indexers.faiss_indexer import FaissIndexer
from src.indexers.azure_search_indexer import AzureSearchIndexer
from src.schemas import Chunk

Strategy = Literal["character","recursive","adaptive"]

def make_chunker(strategy: Strategy):
    if strategy == "character": return CharacterChunker(target_tokens=400, overlap_tokens=40)
    if strategy == "recursive": return RecursiveChunker(max_tokens=500)
    if strategy == "adaptive":  return AdaptiveChunker(max_tokens=550, min_tokens=120, coherence_threshold=0.55)
    raise ValueError(strategy)

def run(strategy: Strategy, batch_size: int = 64):
    setup_logging()
    logger.info(f"Pipeline start strategy='{strategy}'")
    chunker  = make_chunker(strategy)
    embedder = make_embedder()

    backend = os.getenv("INDEX_BACKEND","faiss").lower()
    if backend == "azure":
        indexer = AzureSearchIndexer(dim=embedder.dim)
        logger.info(f"Using Azure AI Search index: {os.getenv('AZURE_SEARCH_INDEX_NAME')}")
    else:
        name = f"{settings.faiss_index_name}_{strategy}"
        indexer = FaissIndexer(settings.index_dir, name, dim=embedder.dim)
        logger.info(f"Using FAISS index: {name}")

    buffer: List[Chunk] = []
    for doc in load_pdfs(settings.pdf_input_dir):
        if not doc.text: continue
        chunks = chunker.chunk(doc); buffer.extend(chunks)
        while len(buffer) >= batch_size:
            batch, buffer = buffer[:batch_size], buffer[batch_size:]
            vecs = embedder.embed([c.text for c in batch])
            indexer.add(batch, vecs)

    if buffer:
        vecs = embedder.embed([c.text for c in buffer])
        indexer.add(buffer, vecs)

    indexer.persist()
    logger.info("Indexing complete.")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "adaptive")
