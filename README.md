# PDF Indexer: Scalable RAG Ingestion + Retrieval Eval

Ingest 200+ PDFs → extract (OCR fallback) → chunk (character/recursive/adaptive) → embed (local or Azure/OpenAI) → index (FAISS dev or Azure AI Search prod) → evaluate retrieval (P@k / Hit@k / MRR; optional re-rank).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Put PDFs into data/raw_pdfs/

# Build indexes (choose one or all)
python -m src.pipeline character
python -m src.pipeline recursive
python -m src.pipeline adaptive
