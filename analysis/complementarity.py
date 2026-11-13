"""
Complementarity Analysis: Decomposition (E6) vs Reformulation (E7) vs Combined (E8) (plan_v2/07 §5).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _load_per_query(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["qid"]
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


def analyze_complementarity(
    path_e6: Path | str,
    path_e7: Path | str,
    path_e8: Path | str,
    *,
    k: int = 100,
) -> dict[str, Any]:
    """Compute target overlap, unique contributions, and oracle upper bound."""
    e6 = _load_per_query(Path(path_e6))
    e7 = _load_per_query(Path(path_e7))
    e8 = _load_per_query(Path(path_e8))

    common_qids = sorted(set(e6.keys()) & set(e7.keys()) & set(e8.keys()))
    if not common_qids:
        return {"error": "no overlapping queries across E6, E7, and E8"}

    n = len(common_qids)
    only_e6 = []
    only_e7 = []
    both_e6_e7 = []
    neither_e6_e7 = []

    e8_improved_count = 0
    e8_diluted_count = 0

    oracle_mrr_list = []
    oracle_ndcg_list = []
    oracle_r_k_list = []

    for qid in common_qids:
        r6 = e6[qid]["rank"]
        r7 = e7[qid]["rank"]
        r8 = e8[qid]["rank"]

        found6 = r6 <= k
        found7 = r7 <= k
        found8 = r8 <= k

        if found6 and not found7:
            only_e6.append(qid)
        elif found7 and not found6:
            only_e7.append(qid)
        elif found6 and found7:
            both_e6_e7.append(qid)
        else:
            neither_e6_e7.append(qid)

        # Oracle rank: min(r6, r7)
        best_rank = min(r6, r7)
        oracle_mrr_list.append(1.0 / best_rank if best_rank < math.inf else 0.0)
        oracle_ndcg_list.append(1.0 / math.log2(best_rank + 1) if best_rank <= 10 else 0.0)
        oracle_r_k_list.append(1.0 if best_rank <= k else 0.0)

        best_baseline_rank = min(r6, r7)
        if r8 < best_baseline_rank:
            e8_improved_count += 1
        elif r8 > best_baseline_rank:
            e8_diluted_count += 1

    return {
        "n_queries": n,
        "k_threshold": k,
        "found_by_e6_only": len(only_e6),
        "found_by_e7_only": len(only_e7),
        "found_by_both": len(both_e6_e7),
        "found_by_neither": len(neither_e6_e7),
        "oracle_ndcg10": sum(oracle_ndcg_list) / n,
        "oracle_mrr": sum(oracle_mrr_list) / n,
        "oracle_recall_at_k": sum(oracle_r_k_list) / n,
        "e8_improved_over_both": e8_improved_count,
        "e8_diluted_below_both": e8_diluted_count,
    }


def render_complementarity_report(stats: dict[str, Any]) -> str:
    """Format complementarity analysis into GitHub markdown."""
    if "error" in stats:
        return f"Error: {stats['error']}"

    k = stats["k_threshold"]
    n = stats["n_queries"]
    lines = [
        "### Complementarity Analysis (Q6)",
        "",
        f"Evaluated on {n} queries at cutoff $K={k}$:",
        "",
        "| Category | Queries | Fraction | Interpretation |",
        "|---|---:|---:|---|",
        f"| Targets retrieved ONLY by Decomposition (E6) | {stats['found_by_e6_only']} | {stats['found_by_e6_only']/n*100:.1f}% | Unique signal from faceted clues |",
        f"| Targets retrieved ONLY by Reformulation (E7) | {stats['found_by_e7_only']} | {stats['found_by_e7_only']/n*100:.1f}% | Unique signal from holistic rewrite |",
        f"| Targets retrieved by BOTH branches | {stats['found_by_both']} | {stats['found_by_both']/n*100:.1f}% | Concordant queries |",
        f"| Targets MISSED by both | {stats['found_by_neither']} | {stats['found_by_neither']/n*100:.1f}% | Hard / out-of-pool queries |",
        "",
        "#### Oracle Fusion Ceiling (Best of E6 and E7 per query)",
        f"- **Oracle nDCG@10:** `{stats['oracle_ndcg10']:.4f}`",
        f"- **Oracle MRR:** `{stats['oracle_mrr']:.4f}`",
        f"- **Oracle Recall@{k}:** `{stats['oracle_recall_at_k']:.4f}`",
        f"- **E8 Improved Ranks:** `{stats['e8_improved_over_both']}` queries",
        f"- **E8 Diluted Ranks:** `{stats['e8_diluted_below_both']}` queries",
    ]
    return "\n".join(lines) + "\n"
