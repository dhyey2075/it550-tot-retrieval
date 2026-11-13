#!/usr/bin/env python3
"""
Run Dev1 decomposition experiments: E4, E5-R, E6, E4c (Plan 05).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_run import main as eval_main
from retrieval.bm25.search import PyseriniSearcher
from retrieval.corpus_lookup import JsonlCorpusCache, PyseriniDocStore
from retrieval.dense.rerank import build_dense_reranker
from retrieval.pipeline_e4e6 import (
    load_cached_facets,
    run_e4,
    run_e4c,
    run_e5_r,
    run_e6,
)
from retrieval.queries import load_queries
from tot.run_naming import make_run_name


def find_index(parent: Path) -> Path:
    marker = parent / "INDEX_READY"
    if marker.exists():
        p = Path(marker.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    cands = [p for p in parent.rglob("*") if p.is_dir() and any(p.glob("segments*"))]
    if not cands:
        raise FileNotFoundError(f"No Lucene index under {parent}")
    return max(cands, key=lambda p: sum(1 for _ in p.glob("*")))


def main() -> int:
    index_parent = ROOT / "indices" / "anserini"
    index_dir = find_index(index_parent)
    split = "dev1"
    queries = load_queries(ROOT / "data" / f"{split}-queries.jsonl")
    print(f"[progress] loaded {len(queries)} {split} queries", flush=True)

    facets_cache = ROOT / "cache" / f"facets_{split}_extractive.jsonl"
    if not facets_cache.exists():
        facets_cache = ROOT / "cache" / "decompositions.jsonl"
    if not facets_cache.exists():
        raise FileNotFoundError(f"Missing facets cache at {facets_cache}")
    facets = load_cached_facets(facets_cache)
    print(f"[progress] loaded facets for {len(facets)} queries from {facets_cache.name}", flush=True)

    bm25 = PyseriniSearcher(index_dir)
    cand_jsonl = ROOT / "data" / "corpus" / "dev1_e1_candidates.jsonl"
    if cand_jsonl.exists() and cand_jsonl.stat().st_size > 0:
        doc_store = JsonlCorpusCache(cand_jsonl, lead_chars=1000)
    else:
        doc_store = PyseriniDocStore(bm25.searcher, lead_chars=1000)

    print("[progress] initializing e5-small reranker on MPS...", flush=True)
    reranker = build_dense_reranker("e5-small", device="mps")

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "corpus_tier": "dense_rerank_bm25",
        "bm25": {"backend": "pyserini", "k1": None, "b": None, "lead_chars": 1000},
        "dense": {"preset": "e5-small", "role": "rerank_bm25_candidates"},
        "rrf_k": 60,
        "prompt_version": "decomposition_extractive_v1",
    }

    # E4: BM25 per-facet + RRF
    t0 = time.time()
    e4_name = make_run_name("E4", split, "tierB", "extractive", date)
    e4_path = runs_dir / f"{e4_name}.run"
    print(f"[progress] running E4 → {e4_path.name}", flush=True)
    run_e4(queries, facets, bm25, out_path=e4_path, run_name=e4_name, rrf_k=60, depth=1000, config=cfg)
    print(f"[progress] E4 completed in {time.time()-t0:.1f}s", flush=True)

    # E4c: Concatenation baseline
    t1 = time.time()
    e4c_name = make_run_name("E4c", split, "tierB", "concat", date)
    e4c_path = runs_dir / f"{e4c_name}.run"
    print(f"[progress] running E4c → {e4c_path.name}", flush=True)
    run_e4c(queries, facets, bm25, out_path=e4c_path, run_name=e4c_name, depth=1000)
    print(f"[progress] E4c completed in {time.time()-t1:.1f}s", flush=True)

    # E5-R: Per-facet BM25 → dense rerank → RRF
    t2 = time.time()
    e5_name = make_run_name("E5-R", split, "tierB", "extractive", date)
    e5_path = runs_dir / f"{e5_name}.run"
    print(f"[progress] running E5-R → {e5_path.name}", flush=True)
    run_e5_r(queries, facets, bm25, reranker, doc_store.text, out_path=e5_path, run_name=e5_name, rrf_k=60, depth=1000, config=cfg)
    print(f"[progress] E5-R completed in {time.time()-t2:.1f}s", flush=True)

    # E6: Hybrid decomposition (BM25 + dense rerank all facets fused via RRF)
    t3 = time.time()
    e6_name = make_run_name("E6", split, "tierB", "extractive", date)
    e6_path = runs_dir / f"{e6_name}.run"
    print(f"[progress] running E6 → {e6_path.name}", flush=True)
    run_e6(queries, facets, bm25, reranker, doc_store.text, out_path=e6_path, run_name=e6_name, rrf_k=60, depth=1000, config=cfg)
    print(f"[progress] E6 completed in {time.time()-t3:.1f}s", flush=True)

    # Evaluate all runs
    qrels = ROOT / "data" / f"{split}-qrels.txt"
    runs_to_eval = [
        ("E4", e4_path),
        ("E4c", e4c_path),
        ("E5-R", e5_path),
        ("E6", e6_path),
    ]
    for exp, path in runs_to_eval:
        print(f"[progress] evaluating {exp} ({path.name})", flush=True)
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

    print("[progress] Dev1 E4/E4c/E5-R/E6 completed and evaluated successfully!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
