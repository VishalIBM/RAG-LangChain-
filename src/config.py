from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    pdf_input_dir: Path = Field(default=Path("./data/raw_pdfs"))
    cache_dir: Path = Field(default=Path("./data/cache"))
    index_dir: Path = Field(default=Path("./data/indexes"))

    # Embeddings
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_api_type: Optional[str] = None
    openai_api_version: Optional[str] = None
    openai_deployment_name: Optional[str] = None

    # Index names
    faiss_index_name: str = "pdf_index_f3"

    # pgvector
    pgvectoR_conn_str: Optional[str] = None
    PGVECTOR_CONN_STR: Optional[str] = None
    PGVECTOR_TABLE: Optional[str] = "pdf_chunks"

    # Extraction & OCR
    extractor_order: str = "pypdf,pymupdf,ocr"
    max_workers: int = 4
    empty_page_min_chars: int = 30
    ocr_lang: str = "eng"
    ocr_render_dpi: int = 220

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
