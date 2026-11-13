"""
Own metric implementations for single-relevant known-item retrieval.

Definitions (plan_v2/03 §2), with list truncated at DEPTH=1000:
- rank r is 1-based; absent → metrics 0 (MRR uses the truncated list only)
- MRR = 1/r
- nDCG@10 = 1/log2(r+1) if r ≤ 10 else 0
- Recall@K = 1 if r ≤ K else 0  (= Success@K here)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from evaluation.io_utils import DEPTH, load_qrels, load_run, normalize_docid, normalize_qid

# Re-export for callers
__all__ = [
    "DEPTH",
    "PerQueryResult",
    "AggregateResult",
    "rank_of_target",
    "per_query_metrics",
    "aggregate_metrics",
    "evaluate_run",
    "bootstrap_ci",
    "map_at_10",
]


@dataclass
class PerQueryResult:
    qid: str
    target_docid: str
    rank_of_target: int | None  # None if not found within DEPTH
    rr: float
    ndcg10: float
    r10: float
    r100: float
    r1000: float
    map10: float  # cut-off AP@10; equals rr if rank≤10 else 0 (paper-comparability)

    def to_row(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "target_docid": self.target_docid,
            "rank_of_target": self.rank_of_target if self.rank_of_target is not None else "",
            "rr": self.rr,
            "ndcg10": self.ndcg10,
            "r10": self.r10,
            "r100": self.r100,
            "r1000": self.r1000,
            "map10": self.map10,
        }


@dataclass
class AggregateResult:
    n_queries: int
    n_missing_from_run: int
    ndcg10: float
    mrr: float
    r10: float
    r100: float
    r1000: float
    map10: float
    not_found_rate: float
    median_rank_found: float | None
    mrr_found_only: float | None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def rank_of_target(ranked_docids: list[str], target_docid: str) -> int | None:
    """1-based rank within the (already truncated) list, or None if absent."""
    target = normalize_docid(target_docid)
    for i, docid in enumerate(ranked_docids, start=1):
        if normalize_docid(docid) == target:
            return i
    return None


def _metrics_from_rank(r: int | None) -> dict[str, float]:
    if r is None:
        return {"rr": 0.0, "ndcg10": 0.0, "r10": 0.0, "r100": 0.0, "r1000": 0.0, "map10": 0.0}
    rr = 1.0 / r
    ndcg10 = (1.0 / math.log2(r + 1)) if r <= 10 else 0.0
    r10 = 1.0 if r <= 10 else 0.0
    r100 = 1.0 if r <= 100 else 0.0
    r1000 = 1.0 if r <= 1000 else 0.0
    map10 = rr if r <= 10 else 0.0
    return {"rr": rr, "ndcg10": ndcg10, "r10": r10, "r100": r100, "r1000": r1000, "map10": map10}


def per_query_metrics(
    rankings: dict[str, list[str]],
    qrels: dict[str, str],
    *,
    depth: int = DEPTH,
) -> tuple[list[PerQueryResult], int]:
    """
    Evaluate every qid in `qrels` (the evaluation set).

    Returns (per_query_list in qrels key order, n_missing_from_run).
    Missing queries count as rank=None / all zeros.
    """
    results: list[PerQueryResult] = []
    missing = 0
    for qid in qrels:
        qid_n = normalize_qid(qid)
        target = qrels[qid]
        if qid_n not in rankings and qid not in rankings:
            missing += 1
            ranked: list[str] = []
        else:
            ranked = rankings.get(qid_n, rankings.get(qid, []))[:depth]
        r = rank_of_target(ranked, target)
        m = _metrics_from_rank(r)
        results.append(
            PerQueryResult(
                qid=qid_n,
                target_docid=normalize_docid(target),
                rank_of_target=r,
                rr=m["rr"],
                ndcg10=m["ndcg10"],
                r10=m["r10"],
                r100=m["r100"],
                r1000=m["r1000"],
                map10=m["map10"],
            )
        )
    return results, missing


def aggregate_metrics(per_query: list[PerQueryResult], *, n_missing_from_run: int = 0) -> AggregateResult:
    if not per_query:
        raise ValueError("per_query is empty")
    n = len(per_query)
    found_ranks = [pq.rank_of_target for pq in per_query if pq.rank_of_target is not None]
    found_rr = [pq.rr for pq in per_query if pq.rank_of_target is not None]
    return AggregateResult(
        n_queries=n,
        n_missing_from_run=n_missing_from_run,
        ndcg10=mean(pq.ndcg10 for pq in per_query),
        mrr=mean(pq.rr for pq in per_query),
        r10=mean(pq.r10 for pq in per_query),
        r100=mean(pq.r100 for pq in per_query),
        r1000=mean(pq.r1000 for pq in per_query),
        map10=mean(pq.map10 for pq in per_query),
        not_found_rate=1.0 - (len(found_ranks) / n),
        median_rank_found=float(median(found_ranks)) if found_ranks else None,
        mrr_found_only=mean(found_rr) if found_rr else None,
    )


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`."""
    import random

    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo_i = int(math.floor((alpha / 2) * n_boot))
    hi_i = int(math.ceil((1 - alpha / 2) * n_boot)) - 1
    hi_i = min(max(hi_i, 0), n_boot - 1)
    return means[lo_i], means[hi_i]


def map_at_10(rank: int | None) -> float:
    """AP@10 with one relevant doc (= 1/r if r≤10 else 0)."""
    return _metrics_from_rank(rank)["map10"]


def evaluate_run(
    run_path: str | Path,
    qrels_path: str | Path,
    *,
    depth: int = DEPTH,
) -> tuple[list[PerQueryResult], AggregateResult]:
    qrels = load_qrels(qrels_path)
    rankings = load_run(run_path, depth=depth)
    per_q, missing = per_query_metrics(rankings, qrels, depth=depth)
    agg = aggregate_metrics(per_q, n_missing_from_run=missing)
    return per_q, agg
