#!/usr/bin/env python3
"""Run Dev1 combined fusion experiments: E8, protocols F1/F2/F4, and complementarity (Plan 07)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.complementarity import analyze_complementarity, render_complementarity_report
from evaluation.evaluate_run import main as eval_main
from retrieval.bm25.search import PyseriniSearcher
from retrieval.corpus_lookup import JsonlCorpusCache, PyseriniDocStore
from retrieval.dense.rerank import build_dense_reranker
from retrieval.hybrid.e8_fusion import run_e8_suite, save_candidate_pool_for_reranking
from retrieval.pipeline_e4e6 import load_cached_facets
from retrieval.queries import load_queries
from tot.reformulation import load_cached_reformulations


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


def find_latest_per_query(per_query_dir: Path, exp_prefix: str) -> Path | None:
    cands = sorted(per_query_dir.glob(f"{exp_prefix}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> int:
    split = "dev1"
    qrels_path = ROOT / "data" / f"{split}-qrels.txt"
    index_dir = find_index(ROOT / "indices" / "anserini")
    queries = load_queries(ROOT / "data" / f"{split}-queries.jsonl")
    print(f"[progress] loaded {len(queries)} {split} queries", flush=True)

    # 1. Load facets
    facets_cache = ROOT / "cache" / f"facets_{split}_extractive.jsonl"
    if not facets_cache.exists():
        facets_cache = ROOT / "cache" / "decompositions.jsonl"
    if not facets_cache.exists():
        raise FileNotFoundError(f"Missing facets cache at {facets_cache}")
    facets = load_cached_facets(facets_cache)
    print(f"[progress] loaded facets for {len(facets)} queries from {facets_cache.name}", flush=True)

    # 2. Load reformulations
    reforms_cache = ROOT / "cache" / f"reformulations_{split}_keyword.jsonl"
    if not reforms_cache.exists():
        raise FileNotFoundError(f"Missing reformulations cache at {reforms_cache}. Run scripts/run_dev1_e7.py first.")
    reforms = load_cached_reformulations(reforms_cache)
    print(f"[progress] loaded reformulations for {len(reforms)} queries from {reforms_cache.name}", flush=True)

    bm25 = PyseriniSearcher(index_dir)
    cand_jsonl = ROOT / "data" / "corpus" / "dev1_e1_candidates.jsonl"
    if cand_jsonl.exists() and cand_jsonl.stat().st_size > 0:
        doc_store = JsonlCorpusCache(cand_jsonl, lead_chars=1000)
    else:
        doc_store = PyseriniDocStore(bm25.searcher, lead_chars=1000)

    print("[progress] initializing e5-small reranker on MPS/CUDA...", flush=True)
    reranker = build_dense_reranker("e5-small")

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "corpus_tier": "dense_rerank_bm25",
        "bm25": {"backend": "pyserini", "k1": None, "b": None, "lead_chars": 1000},
        "dense": {"preset": "e5-small", "role": "rerank_bm25_candidates"},
        "rrf_k": 60,
        "fusion_protocols": ["f1", "f2", "f4"],
    }

    t0 = time.time()
    print("[progress] running E8 suite (E8 [F2], E8-F1, E8-F4)...", flush=True)
    paths = run_e8_suite(
        ROOT / "data" / f"{split}-queries.jsonl",
        facets,
        reforms,
        bm25,
        reranker,
        doc_store.text,
        runs_dir=runs_dir,
        split=split,
        rrf_k=60,
        depth=1000,
        date=date,
        config=cfg,
    )
    print(f"[progress] E8 suite completed in {time.time()-t0:.1f}s", flush=True)

    per_query_dir = ROOT / "evaluation" / "per_query"
    per_query_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each E8 run
    for exp, path in paths.items():
        if exp == "E8-F2":
            continue  # E8 and E8-F2 point to same run
        print(f"[progress] evaluating {exp} ({path.name})", flush=True)
        eval_main(
            [
                str(path),
                "--exp",
                exp,
                "--qrels",
                str(qrels_path),
                "--split",
                split,
                "--corpus-tier",
                "dense_rerank_bm25",
                "--results",
                str(ROOT / "evaluation" / "results.csv"),
                "--per-query-dir",
                str(per_query_dir),
            ]
        )

    # Export candidate pool for Plan 08 reranking
    e8_pool_path = ROOT / "reranking" / f"candidates_E8_{split}.jsonl"
    print(f"[progress] exporting top-100 candidates for Plan 08 to {e8_pool_path.name}...", flush=True)
    save_candidate_pool_for_reranking(paths["E8"], doc_store.text, e8_pool_path, top_k=100)
    print(f"[progress] candidate pool saved ({e8_pool_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # Complementarity Analysis
    e6_pq = find_latest_per_query(per_query_dir, "E6")
    e7_pq = find_latest_per_query(per_query_dir, "E7")
    e8_pq = find_latest_per_query(per_query_dir, "E8")

    if e6_pq and e7_pq and e8_pq:
        print(f"[progress] running complementarity analysis using {e6_pq.name}, {e7_pq.name}, {e8_pq.name}...", flush=True)
        comp_stats = analyze_complementarity(e6_pq, e7_pq, e8_pq, k=100)
        report = render_complementarity_report(comp_stats)
        out_analysis = ROOT / "analysis" / f"complementarity_{split}.md"
        out_analysis.parent.mkdir(parents=True, exist_ok=True)
        out_analysis.write_text(report, encoding="utf-8")
        print(f"[progress] complementarity report written to {out_analysis}")
        print("\n" + report)
    else:
        print(f"[warning] missing per-query files for complementarity: e6={e6_pq}, e7={e7_pq}, e8={e8_pq}")

    print("[progress] Dev1 E8 suite completed successfully!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
