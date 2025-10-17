import hashlib, re, tiktoken
from pathlib import Path
from typing import Dict

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))

def file_checksum(path: Path, algo: str = "sha256", chunk_size: int = 1_048_576) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk_size), b""):
            h.update(b)
    return h.hexdigest()

def file_fingerprint(path: Path) -> Dict[str, str | int | float]:
    stat = path.stat()
    checksum = file_checksum(path)
    return {"modified_time": stat.st_mtime, "file_size": stat.st_size, "checksum": checksum}

def fingerprint_key(path: Path) -> str:
    fp = file_fingerprint(path)
    src = f"{path.resolve()}::{fp['file_size']}::{fp['modified_time']}::{fp['checksum']}"
    return stable_hash(src)
