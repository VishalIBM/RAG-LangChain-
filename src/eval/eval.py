import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from loguru import logger
from .retriever import Retriever
from .reranker import LocalCrossEncoderReranker, RerankItem
from .eval_dataset import EvalDataset, weakly_generate_eval_from_metadata
from ..config import settings

STRATEGIES = ["character","recursive","adaptive"]

def _p_at_k(g:set[str], got:List[str], k:int)->float:
    return float(len(g.intersection(set(got[:k]))))/max(1, min(k, len(got))) if got else 0.0
def _hit_k(g:set[str], got:List[str], k:int)->float:
    return 1.0 if any(x in set(got[:k]) for x in g) else 0.0
def _mrr(g:set[str], got:List[str])->float:
    for i,x in enumerate(got,1):
        if x in g: return 1.0/i
    return 0.0

def eval_one(strategy:str, ds:EvalDataset, k:int, rerank:bool, pool:int)->pd.DataFrame:
    logger.info(f"Eval strategy='{strategy}' k={k} rerank={rerank}")
    retriever = Retriever(strategy); rr = LocalCrossEncoderReranker() if rerank else None
    rows: List[Dict[str,Any]]=[]
    for ex in ds.items:
        res = retriever.retrieve(ex.question, k=max(k,pool if rerank else k))
        if rerank and res:
            items = [RerankItem(text=r.text, meta=r.metadata) for r in res]
            order = rr.rerank(ex.question, items, top_k=k)
            res = [res[i] for i,_ in order]
        got = [r.chunk_id for r in res]; gold = set(ex.expected_chunk_ids or [])
        rows.append({"qid":ex.qid,"strategy":strategy,"k":k,"rerank":rerank,
                     "P@5":_p_at_k(gold,got,5),"P@10":_p_at_k(gold,got,10),
                     "Hit@5":_hit_k(gold,got,5),"Hit@10":_hit_k(gold,got,10),"MRR":_mrr(gold,got)})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eval_json",type=str,default="")
    ap.add_argument("--auto_from_strategy",choices=STRATEGIES,default="adaptive")
    ap.add_argument("--n_auto",type=int,default=30)
    ap.add_argument("--k",type=int,default=10)
    ap.add_argument("--use_rerank",action="store_true")
    ap.add_argument("--rerank_top_k",type=int,default=25)
    ap.add_argument("--out_csv",type=str,default="./data/indexes/rag_eval_results.csv")
    a=ap.parse_args()

    if a.eval_json:
        ds = EvalDataset(Path(a.eval_json)).load()
        logger.info(f"Loaded eval items: {len(ds.items)}")
    else:
        meta = settings.index_dir / f"{settings.faiss_index_name}_{a.auto_from_strategy}.parquet"
        logger.info("Bootstrapping weak eval set (directional only).")
        ds = weakly_generate_eval_from_metadata(meta, n=a.n_auto)

    frames=[]
    for s in STRATEGIES:
        try: frames.append(eval_one(s, ds, k=a.k, rerank=a.use_rerank, pool=a.rerank_top_k))
        except FileNotFoundError as e: logger.warning(f"Skip {s}: {e}")
    if not frames: return
    report = pd.concat(frames, ignore_index=True)
    summary = report.groupby(["strategy","rerank"]).agg({"P@5":"mean","P@10":"mean","Hit@5":"mean","Hit@10":"mean","MRR":"mean"}).reset_index().sort_values(["rerank","MRR"], ascending=[False,False])
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(a.out_csv, index=False)
    print("\n=== RAG Eval (per-query) ==="); print(report.head(20).to_string(index=False))
    print("\n=== Summary by strategy ==="); print(summary.to_string(index=False))
    print(f"\nSaved: {a.out_csv}")

if __name__=="__main__": main()
