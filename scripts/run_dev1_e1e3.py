#!/usr/bin/env python3
"""Local Dev1 runner: E1 → E2-R → E3 (Tier B) + evaluate."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_run import main as eval_main
from evaluation.io_utils import load_run
from retrieval.bm25.search import PyseriniSearcher
from retrieval.corpus_lookup import JsonlCorpusCache, PyseriniDocStore
from retrieval.dense.rerank import build_dense_reranker
from retrieval.pipeline_e1e3 import run_e1, run_e2_r, run_e3
from retrieval.queries import load_queries
from tot.run_naming import make_run_name


def find_index(parent: Path) -> Path:
    marker = parent / "INDEX_READY"
    if marker.exists():
        p = Path(marker.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    # heuristic
    cands = [p for p in parent.rglob("*") if p.is_dir() and any(p.glob("segments*"))]
    if not cands:
        raise FileNotFoundError(f"No Lucene index under {parent}")
    return max(cands, key=lambda p: sum(1 for _ in p.glob("*")))


def main() -> int:
    index_parent = ROOT / "indices" / "anserini"
    index_dir = find_index(index_parent)
    print(f"[progress] index={index_dir}", flush=True)

    split = "dev1"
    queries = load_queries(ROOT / "data" / f"{split}-queries.jsonl")
    print(f"[progress] loaded {len(queries)} {split} queries", flush=True)

    # D-02 defaults until grid: engine defaults
    bm25 = PyseriniSearcher(index_dir)
    cand_jsonl = ROOT / "data" / "corpus" / "dev1_e1_candidates.jsonl"
    if cand_jsonl.exists() and cand_jsonl.stat().st_size > 0:
        print(f"[progress] loading doc texts from {cand_jsonl}", flush=True)
        doc_store = JsonlCorpusCache(cand_jsonl, lead_chars=1000)
        print(f"[progress] doc cache size={len(doc_store)}", flush=True)
    else:
        print("[progress] WARNING: no candidate JSONL; Pyserini raw() may be empty", flush=True)
        doc_store = PyseriniDocStore(bm25.searcher, lead_chars=1000)
    print("[progress] loading dense model e5-small (MPS if available)...", flush=True)
    reranker = build_dense_reranker("e5-small", device="mps")

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "corpus_tier": "dense_rerank_bm25",
        "bm25": {"backend": "pyserini", "k1": None, "b": None, "lead_chars": 1000},
        "dense": {"preset": "e5-small", "role": "rerank_bm25_candidates"},
        "rrf_k": 60,
    }

    t0 = time.time()
    e1_name = make_run_name("E1", split, "tierB", "orig", date)
    e1_path = runs_dir / f"{e1_name}.run"
    print(f"[progress] E1 BM25 → {e1_path.name}", flush=True)
    run_e1(queries, bm25, out_path=e1_path, run_name=e1_name, k=1000, config=cfg)
    print(f"[progress] E1 done in {time.time()-t0:.1f}s", flush=True)
    bm25_rankings = load_run(e1_path)

    t1 = time.time()
    e2_name = make_run_name("E2-R", split, "tierB", "orig", date)
    e2_path = runs_dir / f"{e2_name}.run"
    print(f"[progress] E2-R dense rerank (142×≤1000 docs) → {e2_path.name}", flush=True)
    # Progress per query
    from evaluation.io_utils import write_trec_run
    from datetime import datetime as dt

    rankings = {}
    scores = {}
    for i, row in enumerate(queries, 1):
        qid = row["qid"]
        cands = bm25_rankings.get(qid, [])[:1000]
        hits = reranker.rerank(row["query"], cands, doc_store.text, top_k=1000)
        rankings[qid] = [d for d, _ in hits]
        scores[qid] = [s for _, s in hits]
        if i % 10 == 0 or i == 1:
            print(f"  E2-R {i}/{len(queries)} elapsed={time.time()-t1:.0f}s", flush=True)
    write_trec_run(e2_path, rankings, run_name=e2_name, scores=scores)
    e2_path.with_suffix(".json").write_text(
        json.dumps({"exp": "E2-R", "config": cfg, "n_queries": len(queries)}, indent=2) + "\n"
    )
    print(f"[progress] E2-R done in {time.time()-t1:.1f}s", flush=True)
    dense_rankings = rankings

    t2 = time.time()
    e3_name = make_run_name("E3", split, "tierB", "orig", date)
    e3_path = runs_dir / f"{e3_name}.run"
    print(f"[progress] E3 RRF → {e3_path.name}", flush=True)
    run_e3(
        queries,
        bm25_rankings,
        dense_rankings,
        out_path=e3_path,
        run_name=e3_name,
        rrf_k=60,
        out_depth=1000,
        config=cfg,
    )
    print(f"[progress] E3 done in {time.time()-t2:.1f}s", flush=True)

    qrels = ROOT / "data" / f"{split}-qrels.txt"
    for exp, path in [("E1", e1_path), ("E2-R", e2_path), ("E3", e3_path)]:
        print(f"[progress] evaluating {exp}", flush=True)
        eval_main(
            [
                str(path),
                "--exp",
                exp,
                "--qrels",
                str(qrels),
                "--split",
                split,
                "--corpus-tier",
                "dense_rerank_bm25",
                "--results",
                str(ROOT / "evaluation" / "results.csv"),
                "--per-query-dir",
                str(ROOT / "evaluation" / "per_query"),
            ]
        )

    print("[progress] Dev1 E1/E2-R/E3 complete", flush=True)
    print(f"total_wall_s={time.time()-t0:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
