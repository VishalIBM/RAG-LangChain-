from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import json, random, pandas as pd

@dataclass
class QAExample:
    qid: str
    question: str
    answer: str
    expected_chunk_ids: List[str] | None = None

class EvalDataset:
    def __init__(self, path: Path): self.path, self.items = path, []
    def load(self):
        data = json.loads(self.path.read_text())
        self.items = [QAExample(**d) for d in data]; return self
    def save(self):
        self.path.write_text(json.dumps([vars(x) for x in self.items], indent=2))
    @staticmethod
    def from_dataframe(df: pd.DataFrame, q_col="question", a_col="answer", id_col="qid"):
        ds = EvalDataset(Path(":memory:")); ds.items = [QAExample(str(r[id_col]), r[q_col], r[a_col]) for _, r in df.iterrows()]; return ds

def weakly_generate_eval_from_metadata(meta_parquet: Path, n: int = 30, seed: int = 13) -> EvalDataset:
    random.seed(seed); df = pd.read_parquet(meta_parquet)
    cands = df[df["text"].str.contains(r"\b(Introduction|Conclusion|Table|Figure|APR|Net|Total|Regulation|Section)\b", case=True, regex=True)]
    cands = cands.sample(min(n, len(cands))) if len(cands) else df.sample(min(n, len(df)))
    items = []
    for i, r in cands.reset_index().iterrows():
        t = str(r["text"])[:600]
        q = "What APR is stated in the section?" if "APR" in t else ("Which regulation is referenced?" if "Regulation" in t else ("What total value is mentioned?" if "Total" in t else "Summarize the key point of this section in one sentence."))
        items.append(QAExample(qid=f"auto_{i}", question=q, answer=t, expected_chunk_ids=[r["chunk_id"]]))
    ds = EvalDataset(Path(":weak_auto:")); ds.items = items; return ds
