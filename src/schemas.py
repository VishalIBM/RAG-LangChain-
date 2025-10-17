from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class Document:
    doc_id: str
    path: str
    page: int
    text: str
    metadata: Dict[str, Any]

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
