#!/usr/bin/env python3
"""
Light Dev1 tune for D-02/D-03/D-04 (coarse grid, freeze best).

Primary objective for BM25: Recall@1000.
Then light lead_chars + RRF-k checks around defaults.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.io_utils import load_qrels, write_trec_run
from evaluation.metrics import aggregate_metrics, per_query_metrics
from retrieval.bm25.search import PyseriniSearcher
from retrieval.corpus_lookup import JsonlCorpusCache
from retrieval.dense.rerank import build_dense_reranker
from retrieval.queries import load_queries
from retrieval.rrf import rrf_fuse


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


def score_rankings(rankings: dict[str, list[str]], qrels: dict[str, str]) -> dict[str, float]:
    per_q, missing = per_query_metrics(rankings, qrels)
    agg = aggregate_metrics(per_q, n_missing_from_run=missing)
    return {
        "ndcg10": agg.ndcg10,
        "mrr": agg.mrr,
        "r10": agg.r10,
        "r100": agg.r100,
        "r1000": agg.r1000,
        "map10": agg.map10,
    }


def run_bm25(queries, searcher, k=1000):
    rankings, scores = {}, {}
    for row in queries:
        hits = searcher.search(row["query"], k=k)
        rankings[row["qid"]] = [d for d, _ in hits]
        scores[row["qid"]] = [s for _, s in hits]
    return rankings, scores


def main() -> int:
    split = "dev1"
    queries = load_queries(ROOT / "data" / f"{split}-queries.jsonl")
    qrels = load_qrels(ROOT / "data" / f"{split}-qrels.txt")
    index_dir = find_index(ROOT / "indices" / "anserini")
    cand_jsonl = ROOT / "data" / "corpus" / "dev1_e1_candidates.jsonl"
    out_dir = ROOT / "logs" / "tune_dev1"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = [
        (0.9, 0.4),
        (1.2, 0.75),
        (1.5, 0.75),
        (2.0, 0.75),
        (0.9, 0.75),
        (1.2, 0.4),
        (1.2, 1.0),
        (2.5, 0.75),
    ]
    print(f"[progress] D-02 BM25 grid n={len(grid)} on {split}", flush=True)
    bm25_rows = []
    best = None
    for k1, b in grid:
        t0 = time.time()
        searcher = PyseriniSearcher(index_dir, k1=k1, b=b)
        rankings, scores = run_bm25(queries, searcher, k=1000)
        metrics = score_rankings(rankings, qrels)
        row = {
            "k1": k1,
            "b": b,
            "r1000": metrics["r1000"],
            "ndcg10": metrics["ndcg10"],
            "mrr": metrics["mrr"],
            "wall_s": round(time.time() - t0, 1),
        }
        bm25_rows.append(row)
        print(
            f"  BM25 k1={k1} b={b} R@1000={row['r1000']:.4f} "
            f"nDCG10={row['ndcg10']:.4f} ({row['wall_s']}s)",
            flush=True,
        )
        key = (row["r1000"], row["ndcg10"], row["mrr"])
        if best is None or key > best[0]:
            best = (key, row, rankings, scores)

    assert best is not None
    best_bm25 = best[1]
    bm25_rankings, bm25_scores = best[2], best[3]
    print(f"[progress] best BM25 {best_bm25}", flush=True)

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    e1_path = ROOT / "runs" / f"E1_dev1_tierB_tuned_{date}.run"
    write_trec_run(e1_path, bm25_rankings, run_name=e1_path.stem, scores=bm25_scores)

    if not cand_jsonl.exists():
        raise SystemExit(f"need {cand_jsonl} for dense tune")
    print("[progress] loading e5-small for D-03", flush=True)
    reranker = build_dense_reranker("e5-small", device="mps")
    lead_grid = [500, 1000, 1500]
    dense_rows = []
    best_dense = None
    for lead in lead_grid:
        t0 = time.time()
        store = JsonlCorpusCache(cand_jsonl, lead_chars=lead)
        rankings, scores = {}, {}
        for i, row in enumerate(queries, 1):
            qid = row["qid"]
            cands = bm25_rankings.get(qid, [])[:1000]
            hits = reranker.rerank(row["query"], cands, store.text, top_k=1000)
            rankings[qid] = [d for d, _ in hits]
            scores[qid] = [s for _, s in hits]
            if i % 40 == 0:
                print(f"    lead={lead} {i}/{len(queries)}", flush=True)
        metrics = score_rankings(rankings, qrels)
        drow = {
            "lead_chars": lead,
            "preset": "e5-small",
            "ndcg10": metrics["ndcg10"],
            "mrr": metrics["mrr"],
            "r1000": metrics["r1000"],
            "wall_s": round(time.time() - t0, 1),
        }
        dense_rows.append(drow)
        print(
            f"  dense lead={lead} nDCG10={drow['ndcg10']:.4f} MRR={drow['mrr']:.4f}",
            flush=True,
        )
        key = (drow["ndcg10"], drow["mrr"])
        if best_dense is None or key > best_dense[0]:
            best_dense = (key, drow, rankings)

    assert best_dense is not None
    best_lead = best_dense[1]
    dense_rankings = best_dense[2]
    print(f"[progress] best dense {best_lead}", flush=True)

    rrf_grid = [40, 60, 80]
    rrf_rows = []
    best_rrf = None
    for rrf_k in rrf_grid:
        rankings = {}
        for qid in bm25_rankings:
            fused = rrf_fuse(
                [bm25_rankings.get(qid, []), dense_rankings.get(qid, [])],
                k=rrf_k,
                out_depth=1000,
            )
            rankings[qid] = [d for d, _ in fused]
        metrics = score_rankings(rankings, qrels)
        rrow = {
            "rrf_k": rrf_k,
            "ndcg10": metrics["ndcg10"],
            "mrr": metrics["mrr"],
            "r1000": metrics["r1000"],
        }
        rrf_rows.append(rrow)
        print(
            f"  RRF k={rrf_k} nDCG10={rrow['ndcg10']:.4f} MRR={rrow['mrr']:.4f}",
            flush=True,
        )
        key = (rrow["ndcg10"], rrow["mrr"])
        if best_rrf is None or key > best_rrf[0]:
            best_rrf = (key, rrow)

    assert best_rrf is not None
    best_rrf_row = best_rrf[1]
    print(f"[progress] best RRF {best_rrf_row}", flush=True)

    summary = {
        "split": split,
        "D-02": best_bm25,
        "D-03": best_lead,
        "D-04": best_rrf_row,
        "bm25_grid": bm25_rows,
        "dense_grid": dense_rows,
        "rrf_grid": rrf_rows,
        "date": date,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "bm25_grid.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bm25_rows[0].keys()))
        w.writeheader()
        w.writerows(bm25_rows)

    freeze_note = out_dir / "freeze_notes.md"
    freeze_note.write_text(
        "\n".join(
            [
                f"# Freeze candidates ({date})",
                "",
                f"- **D-02** BM25: k1={best_bm25['k1']}, b={best_bm25['b']} "
                f"(R@1000={best_bm25['r1000']:.4f}, nDCG@10={best_bm25['ndcg10']:.4f}) on Dev1",
                f"- **D-03** Dense: e5-small, lead_chars={best_lead['lead_chars']} "
                f"(nDCG@10={best_lead['ndcg10']:.4f}) on Dev1",
                f"- **D-04** RRF k={best_rrf_row['rrf_k']} "
                f"(nDCG@10={best_rrf_row['ndcg10']:.4f}) on Dev1",
                "",
            ]
        )
    )
    print(f"[progress] wrote {out_dir}/summary.json", flush=True)
    print(freeze_note.read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
