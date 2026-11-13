"""
Execution pipeline for Cross-Encoder Reranking (Plan 08, Experiment E9).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evaluation.io_utils import load_run, write_trec_run
from reranking.cross_encoder import (
    CrossEncoderReranker,
    format_query_side,
    verify_tail_invariance,
)
from tot.run_naming import make_run_name


def _sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps({**payload, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_candidate_pool(pool_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load JSONL candidate pool produced by save_candidate_pool_for_reranking."""
    pool: dict[str, list[dict[str, Any]]] = {}
    path = Path(pool_path)
    if not path.exists():
        return pool
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                pool[str(row["qid"])] = row.get("candidates", [])
            except Exception:
                continue
    return pool


def run_e9(
    queries: list[dict[str, str]],
    e8_run_path: str | Path,
    candidate_pool_path: str | Path,
    reranker: CrossEncoderReranker,
    *,
    out_path: str | Path,
    run_name: str,
    k: int = 50,
    query_side_format: str = "orig",
    reformulations_by_qid: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """
    Reranks top-K of E8 run with cross-encoder. Tail (> K) preserves exact E8 order.
    Enforces tail invariance assertion check.
    """
    out_path = Path(out_path)
    e8_rankings = load_run(e8_run_path)
    candidate_pool = load_candidate_pool(candidate_pool_path)
    reforms = reformulations_by_qid or {}

    cached_scores = reranker.load_cache(query_side_format)
    e9_rankings: dict[str, list[str]] = {}
    e9_scores: dict[str, list[float]] = {}

    for row in queries:
        qid = str(row["qid"])
        orig_query = row["query"]
        reform_query = reforms.get(qid, "")
        q_text = format_query_side(orig_query, reform_query, style=query_side_format)

        full_e8_list = e8_rankings.get(qid, [])
        # Strictly take top-K docids from the input E8 run to guarantee tail invariance
        top_k_docids = full_e8_list[:k]
        pool_map = {str(c["docid"]): c.get("text", "") for c in candidate_pool.get(qid, [])}
        cands_to_score = [
            {"rank": i, "docid": d, "text": pool_map.get(d, "")}
            for i, d in enumerate(top_k_docids, 1)
        ]

        # Score top-K with cross-encoder
        scored_top_k = reranker.score_and_cache(
            qid,
            q_text,
            cands_to_score,
            input_format=query_side_format,
            top_k=k,
            cached_scores=cached_scores,
        )

        reranked_docids = [docid for docid, _ in scored_top_k]
        reranked_scores = [score for _, score in scored_top_k]

        # Append tail (> K) unchanged from E8 list
        tail_docids = full_e8_list[k:]
        full_reranked_docids = reranked_docids + tail_docids

        # Assign synthetic decreasing scores for tail to maintain TREC ranking order
        min_ce_score = min(reranked_scores) if reranked_scores else 0.0
        tail_scores = [min_ce_score - float(i + 1) for i in range(len(tail_docids))]
        full_scores = reranked_scores + tail_scores

        e9_rankings[qid] = full_reranked_docids
        e9_scores[qid] = full_scores

    # Automated invariance check (Plan 08 §8 & §9)
    ok, detail = verify_tail_invariance(e8_rankings, e9_rankings, k=k)
    if not ok:
        raise AssertionError(f"Tail invariance check failed between E8 and E9: {detail}")

    write_trec_run(out_path, e9_rankings, run_name=run_name, scores=e9_scores)
    _sidecar(
        out_path,
        {
            "exp": "E9",
            "run_name": run_name,
            "reranker_model": reranker.model_name,
            "k_depth": k,
            "query_side_format": query_side_format,
            "config": config or {},
            "tail_invariance_verified": True,
        },
    )
    return out_path


def virtual_tune_k(
    queries: list[dict[str, str]],
    e8_run_path: str | Path,
    candidate_pool_path: str | Path,
    reranker: CrossEncoderReranker,
    qrels_path: str | Path,
    *,
    k_options: Sequence[int] = (20, 30, 50, 100),
    query_side_format: str = "orig",
    reformulations_by_qid: dict[str, str] | None = None,
    runs_dir: str | Path = "runs",
    split: str = "dev1",
    date: str | None = None,
) -> dict[int, dict[str, float]]:
    """
    Virtual tuning (Plan 08 §3.3): scores top-100 once into cache, then
    evaluates K in [20, 30, 50, 100] in memory to find optimal depth K.
    """
    from evaluation.metrics import aggregate_metrics, per_query_metrics
    from evaluation.io_utils import load_qrels

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
    qrels = load_qrels(qrels_path)
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, float]] = {}
    for k in k_options:
        run_name = make_run_name("E9", split, "tierB", f"k{k}_{query_side_format}", date)
        out_path = runs_dir / f"{run_name}.run"
        run_e9(
            queries,
            e8_run_path,
            candidate_pool_path,
            reranker,
            out_path=out_path,
            run_name=run_name,
            k=k,
            query_side_format=query_side_format,
            reformulations_by_qid=reformulations_by_qid,
        )
        rankings = load_run(out_path)
        per_q, missing = per_query_metrics(rankings, qrels)
        agg = aggregate_metrics(per_q, n_missing_from_run=missing)
        results[k] = {
            "ndcg10": agg.ndcg10,
            "mrr": agg.mrr,
            "r10": agg.r10,
            "r100": agg.r100,
            "r1000": agg.r1000,
            "run_path": str(out_path),
        }
    return results
