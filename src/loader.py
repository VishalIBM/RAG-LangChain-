from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple
import json, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
from .schemas import Document
from .utils import stable_hash, clean_text, fingerprint_key, file_fingerprint
from .config import settings

def _extract_with_pypdf(pdf_path: Path) -> Tuple[List[str], Dict[str, Any]]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try: pages.append(p.extract_text() or "")
        except Exception: pages.append("")
    meta = reader.metadata or {}
    title = (meta.get("/Title") or "").strip() or None
    return pages, {"doc_title": title, "num_pages": len(pages), "extractor_used": "pypdf"}

def _extract_with_pymupdf(pdf_path: Path) -> Tuple[List[str], Dict[str, Any]]:
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        pages = [pg.get_text("text") or "" for pg in doc]
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip() or None
        return pages, {"doc_title": title, "num_pages": len(pages), "extractor_used": "pymupdf"}

def _ocr_page_with_pymupdf(pdf_path: Path, page_index: int, dpi: int, lang: str) -> str:
    import fitz, io, pytesseract
    from PIL import Image
    with fitz.open(str(pdf_path)) as doc:
        mat = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = doc.load_page(page_index).get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return clean_text(pytesseract.image_to_string(img, lang=lang) or "")

def _cache_paths(pdf_path: Path, fp_key: str) -> Tuple[Path, Path]:
    base = f"{pdf_path.stem}__{fp_key}"
    return settings.cache_dir / f"{base}.parquet", settings.cache_dir / f"{base}.json"

def _find_existing_cache(pdf_path: Path) -> Tuple[Path | None, Path | None]:
    stem = pdf_path.stem
    cands = sorted(settings.cache_dir.glob(f"{stem}__*.parquet"))
    if not cands: return None, None
    pq = cands[-1]; js = pq.with_suffix(".json")
    return (pq if pq.exists() else None), (js if js.exists() else None)

def _extract_pdf_to_cache(pdf_path_str: str, extractor_order: List[str],
                          empty_min_chars: int, ocr_lang: str, ocr_dpi: int) -> Tuple[str, str | None]:
    pdf_path = Path(pdf_path_str)
    try:
        pages, meta = [], {}
        for name in [e.strip().lower() for e in extractor_order]:
            try:
                if name == "pypdf":
                    pages, meta = _extract_with_pypdf(pdf_path)
                elif name == "pymupdf":
                    pages, meta = _extract_with_pymupdf(pdf_path)
                elif name == "ocr":
                    import fitz
                    with fitz.open(str(pdf_path)) as doc:
                        pages = ["" for _ in range(len(doc))]
                        meta = {"doc_title": (doc.metadata or {}).get("title") or None,
                                "num_pages": len(doc), "extractor_used": "ocr_only"}
                if any(pages): break
            except Exception:
                continue

        if not pages:
            return str(pdf_path), f"Failed to extract using order={extractor_order}"

        fp = file_fingerprint(pdf_path)
        fp_key = fingerprint_key(pdf_path)
        parquet_path, meta_path = _cache_paths(pdf_path, fp_key)

        ocr_used_pages: List[int] = []
        for i, t in enumerate(pages):
            if (t or "").strip() and len(t.strip()) >= empty_min_chars: continue
            try:
                ocr_text = _ocr_page_with_pymupdf(pdf_path, i, dpi=settings.ocr_render_dpi, lang=settings.ocr_lang)
                if len(ocr_text) >= len((t or "").strip()):
                    pages[i] = ocr_text
                    ocr_used_pages.append(i+1)
            except Exception:
                pass

        rows = []
        for i, text in enumerate(pages, start=1):
            doc_key = f"{pdf_path.resolve()}::{i}"
            doc_id = stable_hash(doc_key)
            rows.append({
                "doc_id": doc_id,
                "path": str(pdf_path.resolve()),
                "page": i,
                "text": clean_text(text or ""),
                "metadata": {
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path.resolve()),
                    "page": i,
                    "source": "pdf",
                    "doc_title": meta.get("doc_title"),
                }
            })
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)

        meta_path.write_text(json.dumps({
            "file": str(pdf_path.resolve()),
            "fingerprint": fp,
            "fingerprint_key": fp_key,
            "num_pages": meta.get("num_pages", len(pages)),
            "doc_title": meta.get("doc_title"),
            "extractor_used": meta.get("extractor_used"),
            "ocr_used_pages": ocr_used_pages
        }, indent=2))
        return str(pdf_path), None
    except Exception as e:
        return str(pdf_path), f"{type(e).__name__}: {e}"

def load_pdfs(root: Path | None = None) -> Iterable[Document]:
    root = root or settings.pdf_input_dir
    pdfs = sorted(root.rglob("*.pdf"))
    logger.info(f"Discovered {len(pdfs)} PDFs under {root}")

    extractor_order = [e.strip() for e in settings.extractor_order.split(",") if e.strip()]
    to_process: List[Path] = []
    for pdf in pdfs:
        key = fingerprint_key(pdf)
        pq, js = _cache_paths(pdf, key)
        if not (pq.exists() and js.exists()):
            to_process.append(pdf)

    if to_process:
        logger.info(f"{len(to_process)} PDFs need extraction/refresh. Workers={settings.max_workers}")
        with ProcessPoolExecutor(max_workers=settings.max_workers) as ex:
            futs = {ex.submit(_extract_pdf_to_cache, str(p), extractor_order,
                              settings.empty_page_min_chars, settings.ocr_lang, settings.ocr_render_dpi): p
                    for p in to_process}
            for fut in as_completed(futs):
                pdf_path, err = fut.result()
                logger.info(f"[EXTRACT] {pdf_path}: {'OK' if not err else err}")

    for pdf in pdfs:
        key = fingerprint_key(pdf)
        pq, _ = _cache_paths(pdf, key)
        if not pq.exists():
            pq_old, _ = _find_existing_cache(pdf)
            if not pq_old:
                logger.warning(f"No cache for {pdf.name}; skipping")
                continue
            pq = pq_old
        df = pd.read_parquet(pq)
        for _, r in df.iterrows():
            yield Document(r["doc_id"], r["path"], int(r["page"]), r["text"], r["metadata"])
