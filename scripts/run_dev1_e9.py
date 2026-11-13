#!/usr/bin/env python3
"""Run Dev1 Cross-Encoder reranking experiments: E9, depth tuning, query ablation (Plan 08)."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.reranking_analysis import analyze_reranking, render_reranking_report
from evaluation.evaluate_run import main as eval_main
from evaluation.io_utils import load_run
from reranking.cross_encoder import CrossEncoderReranker
from reranking.pipeline_e9 import run_e9, virtual_tune_k
from retrieval.queries import load_queries
from tot.reformulation import load_cached_reformulations
from tot.run_naming import make_run_name


def find_latest_run(runs_dir: Path, prefix: str, split: str) -> Path:
    cands = sorted(runs_dir.glob(f"{prefix}_{split}_*.run"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No {prefix} run found under {runs_dir} for split {split}")
    return cands[0]


def find_latest_per_query(per_query_dir: Path, exp_prefix: str) -> Path | None:
    cands = sorted(per_query_dir.glob(f"{exp_prefix}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> int:
    split = "dev1"
    qrels_path = ROOT / "data" / f"{split}-qrels.txt"
    runs_dir = ROOT / "runs"
    per_query_dir = ROOT / "evaluation" / "per_query"
    per_query_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(ROOT / "data" / f"{split}-queries.jsonl")
    print(f"[progress] loaded {len(queries)} {split} queries", flush=True)

    # 1. Identify E8 primary input run
    e8_run_path = find_latest_run(runs_dir, "E8", split)
    print(f"[progress] using E8 input run: {e8_run_path.name}", flush=True)

    # 2. Candidate pool
    candidate_pool_path = ROOT / "reranking" / f"candidates_E8_{split}.jsonl"
    if not candidate_pool_path.exists():
        raise FileNotFoundError(f"Missing candidate pool at {candidate_pool_path}. Run Plan 07 first.")
    print(f"[progress] loaded candidate pool ({candidate_pool_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # 3. Reformulations for query-side ablation
    reforms_path = ROOT / "cache" / f"reformulations_{split}_keyword.jsonl"
    reforms = load_cached_reformulations(reforms_path) if reforms_path.exists() else {}

    # 4. Initialize Cross-Encoder
    model_name = "minilm"
    print(f"[progress] initializing CrossEncoderReranker ({model_name})...", flush=True)
    reranker = CrossEncoderReranker(
        model_name=model_name,
        batch_size=64,
        cache_dir=ROOT / "cache" / "ce_scores",
    )
    print(f"[progress] reranker device: {reranker.device.upper()} (is_stub={reranker.is_stub})", flush=True)

    date = datetime.now(timezone.utc).strftime("%Y%m%d")

    # 5. Virtual tuning over depths K in (20, 30, 50, 100) on Dev1 (Plan 08 §3.3)
    print("\n--- Depth K Virtual Tuning on Dev1 ---", flush=True)
    t0 = time.time()
    tune_results = virtual_tune_k(
        queries,
        e8_run_path,
        candidate_pool_path,
        reranker,
        qrels_path,
        k_options=(20, 30, 50, 100),
        query_side_format="orig",
        reformulations_by_qid=reforms,
        runs_dir=runs_dir,
        split=split,
        date=date,
    )
    for k_val, m in sorted(tune_results.items()):
        print(f"  K={k_val:3d} | nDCG@10: {m['ndcg10']:.4f} | MRR: {m['mrr']:.4f} | R@10: {m['r10']:.4f} | R@100: {m['r100']:.4f}")
    print(f"[progress] virtual depth tuning finished in {time.time()-t0:.1f}s", flush=True)

    # Pick optimal K (choose K with highest nDCG@10, prefer smaller K if within noise)
    best_k = max(tune_results.keys(), key=lambda k: (tune_results[k]["ndcg10"], -k))
    print(f"[decision] Optimal depth K={best_k} chosen on Dev1 (frozen for D-09)")

    # 6. Execute Primary E9 Run
    e9_name = make_run_name("E9", split, "tierB", f"k{best_k}_orig", date)
    e9_path = runs_dir / f"{e9_name}.run"
    e9_cfg = {
        "corpus_tier": "dense_rerank_bm25",
        "reranker_model": reranker.model_name,
        "k_depth": best_k,
        "query_side_format": "orig",
        "is_stub": reranker.is_stub,
    }
    print(f"[progress] executing primary E9 run -> {e9_path.name} (K={best_k})...", flush=True)
    run_e9(
        queries,
        e8_run_path,
        candidate_pool_path,
        reranker,
        out_path=e9_path,
        run_name=e9_name,
        k=best_k,
        query_side_format="orig",
        reformulations_by_qid=reforms,
        config=e9_cfg,
    )

    # Evaluate Primary E9
    print(f"[progress] evaluating E9...", flush=True)
    eval_main(
        [
            str(e9_path),
            "--exp",
            "E9",
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

    # 7. Query-Side Ablation on Dev1 (orig vs reform vs concat)
    print("\n--- Query-Side Ablation on Dev1 ---", flush=True)
    for q_side in ["reform", "concat"]:
        ablation_name = make_run_name("E9", split, "tierB", f"k{best_k}_{q_side}", date)
        ablation_path = runs_dir / f"{ablation_name}.run"
        run_e9(
            queries,
            e8_run_path,
            candidate_pool_path,
            reranker,
            out_path=ablation_path,
            run_name=ablation_name,
            k=best_k,
            query_side_format=q_side,
            reformulations_by_qid=reforms,
        )
        exp_tag = f"E9-{q_side}"
        eval_main(
            [
                str(ablation_path),
                "--exp",
                exp_tag,
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

    # 8. Complementarity & Reachable-Fraction Analysis (Plan 08 §5)
    e8_pq = find_latest_per_query(per_query_dir, "E8")
    e9_pq = find_latest_per_query(per_query_dir, "E9")
    if e8_pq and e9_pq:
        print(f"\n[progress] running reranking analysis ({e8_pq.name} vs {e9_pq.name})...", flush=True)
        ce_stats = analyze_reranking(e8_pq, e9_pq, k=best_k)
        rep = render_reranking_report(ce_stats)
        out_analysis = ROOT / "analysis" / f"reranking_analysis_{split}.md"
        out_analysis.write_text(rep, encoding="utf-8")
        print(f"[progress] reranking analysis written to {out_analysis}")
        print("\n" + rep)

    print("[progress] Dev1 Plan 08 completed successfully!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
