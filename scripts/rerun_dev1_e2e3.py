#!/usr/bin/env python3
"""Re-run Dev1 E2-R + E3 using extracted candidate JSONL (skip E1 if present)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_run import main as eval_main
from evaluation.io_utils import load_run, write_trec_run
from retrieval.bm25.search import PyseriniSearcher
from retrieval.corpus_lookup import JsonlCorpusCache
from retrieval.dense.rerank import build_dense_reranker
from retrieval.pipeline_e1e3 import run_e3
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
    cand_jsonl = ROOT / "data" / "corpus" / "dev1_e1_candidates.jsonl"
    if not cand_jsonl.exists():
        raise SystemExit(f"missing {cand_jsonl}")

    e1_path = ROOT / "runs" / "E1_dev1_tierB_orig_20261112.run"
    if not e1_path.exists():
        raise SystemExit(f"missing E1 run {e1_path}")

    split = "dev1"
    queries = load_queries(ROOT / "data" / f"{split}-queries.jsonl")
    print(f"[progress] queries={len(queries)}", flush=True)

    print(f"[progress] loading doc cache {cand_jsonl}", flush=True)
    t_load = time.time()
    doc_store = JsonlCorpusCache(cand_jsonl, lead_chars=1000)
    print(f"[progress] doc cache size={len(doc_store)} in {time.time()-t_load:.1f}s", flush=True)

    # quick non-empty text check
    bm25_rankings = load_run(e1_path)
    sample_qid = queries[0]["qid"]
    sample_docs = bm25_rankings.get(sample_qid, [])[:5]
    nonempty = sum(1 for d in sample_docs if doc_store.text(d).strip())
    print(f"[progress] sample nonempty texts {nonempty}/{len(sample_docs)}", flush=True)
    if nonempty == 0:
        raise SystemExit("doc texts still empty — aborting")

    print("[progress] loading e5-small (mps)", flush=True)
    reranker = build_dense_reranker("e5-small", device="mps")

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    cfg = {
        "corpus_tier": "dense_rerank_bm25",
        "bm25": {"backend": "pyserini", "k1": None, "b": None, "lead_chars": 1000},
        "dense": {"preset": "e5-small", "role": "rerank_bm25_candidates"},
        "rrf_k": 60,
        "doc_source": str(cand_jsonl),
    }

    t1 = time.time()
    e2_name = make_run_name("E2-R", split, "tierB", "orig", date)
    e2_path = ROOT / "runs" / f"{e2_name}.run"
    print(f"[progress] E2-R → {e2_path.name}", flush=True)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    empty_text = 0
    for i, row in enumerate(queries, 1):
        qid = row["qid"]
        cands = bm25_rankings.get(qid, [])[:1000]
        for d in cands:
            if not doc_store.text(d).strip():
                empty_text += 1
        hits = reranker.rerank(row["query"], cands, doc_store.text, top_k=1000)
        rankings[qid] = [d for d, _ in hits]
        scores[qid] = [s for _, s in hits]
        if i % 10 == 0 or i == 1:
            print(f"  E2-R {i}/{len(queries)} elapsed={time.time()-t1:.0f}s", flush=True)
    write_trec_run(e2_path, rankings, run_name=e2_name, scores=scores)
    e2_path.with_suffix(".json").write_text(
        json.dumps({"exp": "E2-R", "config": cfg, "empty_text_lookups": empty_text}, indent=2)
        + "\n"
    )
    print(f"[progress] E2-R done in {time.time()-t1:.1f}s empty_lookups={empty_text}", flush=True)

    t2 = time.time()
    e3_name = make_run_name("E3", split, "tierB", "orig", date)
    e3_path = ROOT / "runs" / f"{e3_name}.run"
    print(f"[progress] E3 RRF → {e3_path.name}", flush=True)
    run_e3(
        queries,
        bm25_rankings,
        rankings,
        out_path=e3_path,
        run_name=e3_name,
        rrf_k=60,
        out_depth=1000,
        config=cfg,
    )
    print(f"[progress] E3 done in {time.time()-t2:.1f}s", flush=True)

    qrels = ROOT / "data" / f"{split}-qrels.txt"
    for exp, path in [("E2-R", e2_path), ("E3", e3_path)]:
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

    print("[progress] Dev1 E2-R/E3 re-run complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
