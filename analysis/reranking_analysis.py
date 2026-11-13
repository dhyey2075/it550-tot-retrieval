"""
Cross-Encoder Reranking Analysis (Plan 08 §5).

Computes:
1. Reachable fraction (share of queries where target is in top-K of E8).
2. Performance conditioned on reachable queries (isolates reranker strength from recall ceiling).
3. Rank movement: promoted, demoted, unchanged queries.
4. Render markdown report for D-09 decision.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from evaluation.io_utils import load_qrels, load_run


def _load_per_query(path: Path | str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["qid"])
            r_str = row.get("rank_of_target", "")
            rank = float(r_str) if r_str and r_str.lower() != "none" else math.inf
            out[qid] = {
                "rank": rank,
                "ndcg10": float(row.get("ndcg10", 0.0)),
                "rr": float(row.get("rr", 0.0)),
                "r10": float(row.get("r10", 0.0)),
                "r100": float(row.get("r100", 0.0)),
                "r1000": float(row.get("r1000", 0.0)),
            }
    return out


def analyze_reranking(
    e8_per_query_path: Path | str,
    e9_per_query_path: Path | str,
    *,
    k: int = 50,
) -> dict[str, Any]:
    """Compute reachable fraction, rank movements, and conditioned metrics."""
    e8_pq = _load_per_query(e8_per_query_path)
    e9_pq = _load_per_query(e9_per_query_path)

    common_qids = sorted(set(e8_pq.keys()) & set(e9_pq.keys()))
    if not common_qids:
        return {"error": "no common queries found between E8 and E9 per-query files"}

    n = len(common_qids)
    reachable_qids = []
    unreachable_qids = []

    promoted_qids = []
    demoted_qids = []
    unchanged_qids = []

    rank_diffs = []

    for qid in common_qids:
        r8 = e8_pq[qid]["rank"]
        r9 = e9_pq[qid]["rank"]

        is_reachable = r8 <= k
        if is_reachable:
            reachable_qids.append(qid)
        else:
            unreachable_qids.append(qid)

        if r9 < r8:
            promoted_qids.append((qid, r8, r9))
        elif r9 > r8:
            demoted_qids.append((qid, r8, r9))
        else:
            unchanged_qids.append(qid)

        if r8 < math.inf and r9 < math.inf:
            rank_diffs.append(r8 - r9)  # positive = improved rank

    # Conditioned metrics on reachable subset
    n_reach = len(reachable_qids)
    if n_reach > 0:
        reach_ndcg10_e8 = sum(e8_pq[q]["ndcg10"] for q in reachable_qids) / n_reach
        reach_ndcg10_e9 = sum(e9_pq[q]["ndcg10"] for q in reachable_qids) / n_reach
        reach_mrr_e8 = sum(e8_pq[q]["rr"] for q in reachable_qids) / n_reach
        reach_mrr_e9 = sum(e9_pq[q]["rr"] for q in reachable_qids) / n_reach
    else:
        reach_ndcg10_e8 = reach_ndcg10_e9 = reach_mrr_e8 = reach_mrr_e9 = 0.0

    return {
        "n_queries": n,
        "k_depth": k,
        "n_reachable": n_reach,
        "reachable_fraction": n_reach / n,
        "promoted_count": len(promoted_qids),
        "demoted_count": len(demoted_qids),
        "unchanged_count": len(unchanged_qids),
        "mean_rank_improvement_reachable": (sum(rank_diffs) / len(rank_diffs)) if rank_diffs else 0.0,
        "overall_ndcg10_e8": sum(e8_pq[q]["ndcg10"] for q in common_qids) / n,
        "overall_ndcg10_e9": sum(e9_pq[q]["ndcg10"] for q in common_qids) / n,
        "overall_mrr_e8": sum(e8_pq[q]["rr"] for q in common_qids) / n,
        "overall_mrr_e9": sum(e9_pq[q]["rr"] for q in common_qids) / n,
        "reachable_ndcg10_e8": reach_ndcg10_e8,
        "reachable_ndcg10_e9": reach_ndcg10_e9,
        "reachable_mrr_e8": reach_mrr_e8,
        "reachable_mrr_e9": reach_mrr_e9,
        "sample_promoted": promoted_qids[:5],
        "sample_demoted": demoted_qids[:5],
    }


def render_reranking_report(stats: dict[str, Any]) -> str:
    """Render markdown report for D-09 decision and Plan 08 analysis."""
    if "error" in stats:
        return f"Error: {stats['error']}"

    n = stats["n_queries"]
    k = stats["k_depth"]
    n_reach = stats["n_reachable"]
    frac_reach = stats["reachable_fraction"]

    lines = [
        f"### Cross-Encoder Reranking Analysis (Plan 08, K={k})",
        "",
        "#### 1. Reachable Fraction (Ceiling Analysis)",
        f"- **Reachable Queries (target in E8 top-{k}):** `{n_reach}/{n}` ({frac_reach*100:.1f}%)",
        f"- **Unreachable Ceiling:** `{n - n_reach}/{n}` ({(1 - frac_reach)*100:.1f}%) queries cannot be solved by any reranker at depth $K={k}$.",
        "",
        "#### 2. Headline Performance Comparison (E8 vs E9)",
        "",
        "| Metric | Full Dev1 (n=142) E8 | Full Dev1 (n=142) E9 | Reachable Subset E8 | Reachable Subset E9 | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
        f"| **nDCG@10** | `{stats['overall_ndcg10_e8']:.4f}` | `{stats['overall_ndcg10_e9']:.4f}` | `{stats['reachable_ndcg10_e8']:.4f}` | `{stats['reachable_ndcg10_e9']:.4f}` | Cross-encoder precision inside top-10 |",
        f"| **MRR** | `{stats['overall_mrr_e8']:.4f}` | `{stats['overall_mrr_e9']:.4f}` | `{stats['reachable_mrr_e8']:.4f}` | `{stats['reachable_mrr_e9']:.4f}` | Reciprocal rank of first target |",
        "",
        "#### 3. Rank Movement Summary",
        f"- **Promoted (improved rank):** `{stats['promoted_count']}` queries ({stats['promoted_count']/n*100:.1f}%)",
        f"- **Demoted (degraded rank):** `{stats['demoted_count']}` queries ({stats['demoted_count']/n*100:.1f}%)",
        f"- **Unchanged:** `{stats['unchanged_count']}` queries ({stats['unchanged_count']/n*100:.1f}%)",
        "",
        "#### 4. Sample Rank Shifts",
    ]
    if stats["sample_promoted"]:
        lines.append("- **Promoted Samples:**")
        for qid, r8, r9 in stats["sample_promoted"]:
            lines.append(f"  - Query `{qid}`: Rank `{int(r8)}` → `{int(r9)}` (+{int(r8-r9)})")
    if stats["sample_demoted"]:
        lines.append("- **Demoted Samples:**")
        for qid, r8, r9 in stats["sample_demoted"]:
            lines.append(f"  - Query `{qid}`: Rank `{int(r8)}` → `{int(r9)}` ({int(r8-r9)})")

    return "\n".join(lines) + "\n"
