"""Optional ranx cross-check (plan_v2/03 §3.1)."""

from __future__ import annotations

from typing import Any

from evaluation.metrics import AggregateResult, PerQueryResult, aggregate_metrics, per_query_metrics


def ranx_available() -> bool:
    try:
        import ranx  # noqa: F401

        return True
    except ImportError:
        return False


def aggregate_with_ranx(
    rankings: dict[str, list[str]],
    qrels: dict[str, str],
) -> dict[str, float]:
    """
    Compute headline metrics with ranx on the same truncated rankings.

    Returns dict with ndcg@10, mrr, recall@10/100/1000.
    """
    from ranx import Qrels, Run, evaluate

    # ranx qrels: {qid: {docid: rel}}
    qrels_dict = {qid: {docid: 1} for qid, docid in qrels.items()}
    # Ensure every qid appears in the run (missing → empty list → score 0)
    run_dict: dict[str, dict[str, float]] = {}
    for qid in qrels:
        docs = rankings.get(qid, [])
        # scores descending by rank position
        run_dict[qid] = {docid: float(len(docs) - i) for i, docid in enumerate(docs)}

    q = Qrels(qrels_dict)
    r = Run(run_dict)
    metrics = ["ndcg@10", "mrr", "recall@10", "recall@100", "recall@1000"]
    scores = evaluate(q, r, metrics)
    # ranx may return a dict or a float depending on version / single metric
    if isinstance(scores, dict):
        return {k: float(v) for k, v in scores.items()}
    raise RuntimeError(f"Unexpected ranx evaluate return type: {type(scores)}")


def agree_with_ranx(
    rankings: dict[str, list[str]],
    qrels: dict[str, str],
    *,
    tol: float = 1e-6,
) -> tuple[bool, dict[str, Any]]:
    per_q, missing = per_query_metrics(rankings, qrels)
    own = aggregate_metrics(per_q, n_missing_from_run=missing)
    lib = aggregate_with_ranx(rankings, qrels)

    mapping = {
        "ndcg10": "ndcg@10",
        "mrr": "mrr",
        "r10": "recall@10",
        "r100": "recall@100",
        "r1000": "recall@1000",
    }
    diffs = {}
    ok = True
    for own_k, ranx_k in mapping.items():
        a = getattr(own, own_k)
        b = float(lib[ranx_k])
        diff = abs(a - b)
        diffs[own_k] = {"own": a, "ranx": b, "abs_diff": diff}
        if diff > tol:
            ok = False
    return ok, {"missing": missing, "diffs": diffs, "own": own, "ranx": lib}
