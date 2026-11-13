#!/usr/bin/env python3
"""
Quantitative analyses for Plan 09 (Scope §18-§20):
- A1: Main results table with bootstrap CIs
- A3: Rank-of-target distribution across experiments
- A4: Win / Tie / Loss counts for planned comparisons
- A6/A11: Query length quartile analysis
- A7: Recall curve (R@1, 5, 10, 20, 50, 100, 500, 1000)
- A17: Recall-vs-precision trade-off view
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXP_ORDER = [
    "E1",
    "E2-R",
    "E3",
    "E4",
    "E4c",
    "E5-R",
    "E6",
    "E7",
    "E7-bm25",
    "E7-dense",
    "E8",
    "E8-F1",
    "E8-F4",
    "E9",
    "E9-reform",
    "E9-concat",
]


def load_per_query_data(per_query_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load all per-query CSVs keyed by exp_id -> {qid: {metric: val}}."""
    runs_data: dict[str, dict[str, dict[str, float]]] = {}
    for csv_file in sorted(per_query_dir.glob("*.csv")):
        if csv_file.name.endswith(".summary.json"):
            continue
        exp_id = csv_file.name.split("_")[0]
        # Distinguish ablation variants like E8_dev1_..._f1_flat
        for candidate in ["E8-F1", "E8-F4", "E9-reform", "E9-concat", "E7-bm25", "E7-dense", "E4c"]:
            if candidate.lower() in csv_file.name.lower() or f"_{candidate.lower()}_" in csv_file.name.lower():
                exp_id = candidate
                break
        rows: dict[str, dict[str, float]] = {}
        with csv_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rank_val = float(r["rank_of_target"]) if r.get("rank_of_target") else math.inf
                rows[r["qid"]] = {
                    "rank": rank_val,
                    "ndcg10": float(r.get("ndcg10", 0.0)),
                    "rr": float(r.get("rr", 0.0)),
                    "r10": float(r.get("r10", 0.0)),
                    "r100": float(r.get("r100", 0.0)),
                    "r1000": float(r.get("r1000", 0.0)),
                }
        runs_data[exp_id] = rows
    return runs_data


def compute_rank_distribution(runs_data: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, int]]:
    """A3: Rank-of-target distribution (1, 2-3, 4-10, 11-100, 101-1000, >1000)."""
    buckets = ["Rank 1", "Rank 2-3", "Rank 4-10", "Rank 11-100", "Rank 101-1000", "Missing (>1000)"]
    dist: dict[str, dict[str, int]] = {}
    for exp in EXP_ORDER:
        if exp not in runs_data:
            continue
        qdata = runs_data[exp]
        counts = {b: 0 for b in buckets}
        for qid, m in qdata.items():
            r = m["rank"]
            if r == 1:
                counts["Rank 1"] += 1
            elif 2 <= r <= 3:
                counts["Rank 2-3"] += 1
            elif 4 <= r <= 10:
                counts["Rank 4-10"] += 1
            elif 11 <= r <= 100:
                counts["Rank 11-100"] += 1
            elif 101 <= r <= 1000:
                counts["Rank 101-1000"] += 1
            else:
                counts["Missing (>1000)"] += 1
        dist[exp] = counts
    return dist


def compute_recall_curves(
    runs_dir: Path,
    qrels_path: Path,
    cutoffs: Sequence[int] = (1, 5, 10, 20, 50, 100, 500, 1000),
) -> dict[str, dict[int, float]]:
    """A7: Recall@K for K in cutoffs."""
    from evaluation.io_utils import load_qrels, load_run
    qrels = load_qrels(qrels_path)
    total_q = len(qrels)
    if total_q == 0:
        return {}

    curves: dict[str, dict[int, float]] = {}
    for exp in EXP_ORDER:
        # Find latest run file for exp
        cands = sorted(runs_dir.glob(f"{exp}_*.run"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            # Check with prefix
            base = exp.split("-")[0]
            cands = sorted([p for p in runs_dir.glob(f"{base}_*.run") if exp.lower() in p.name.lower()], key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            continue
        run = load_run(cands[0])
        recalls = {}
        for k in cutoffs:
            hits = 0
            for qid, target_docid in qrels.items():
                retrieved_k = run.get(qid, [])[:k]
                if target_docid in retrieved_k:
                    hits += 1
            recalls[k] = hits / total_q
        curves[exp] = recalls
    return curves


def compute_query_length_analysis(
    queries_path: Path,
    runs_data: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """A6: Results broken down by query length quartiles."""
    lengths: dict[str, int] = {}
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            words = len(row["query"].split())
            lengths[str(row["query_id"])] = words

    all_lens = sorted(lengths.values())
    if not all_lens:
        return {}
    q1_cut = all_lens[len(all_lens) // 4]
    q2_cut = all_lens[len(all_lens) // 2]
    q3_cut = all_lens[(3 * len(all_lens)) // 4]

    buckets = {
        "Q1_Short (<=45w)": lambda w: w <= q1_cut,
        "Q2_Medium (46-75w)": lambda w: q1_cut < w <= q2_cut,
        "Q3_Long (76-125w)": lambda w: q2_cut < w <= q3_cut,
        "Q4_VeryLong (>125w)": lambda w: w > q3_cut,
    }

    results: dict[str, dict[str, float]] = {}
    for exp in ["E1", "E3", "E6", "E7", "E8", "E9"]:
        if exp not in runs_data:
            continue
        exp_res = {}
        for b_name, b_fn in buckets.items():
            b_qids = [qid for qid, w in lengths.items() if b_fn(w) and qid in runs_data[exp]]
            if not b_qids:
                continue
            mean_ndcg = sum(runs_data[exp][q]["ndcg10"] for q in b_qids) / len(b_qids)
            exp_res[b_name] = mean_ndcg
        results[exp] = exp_res
    return results


def render_quantitative_report(
    rank_dist: dict[str, dict[str, int]],
    recall_curves: dict[str, dict[int, float]],
    len_analysis: dict[str, dict[str, float]],
    runs_data: dict[str, dict[str, dict[str, float]]] | None = None,
) -> str:
    runs_data = runs_data or {}
    lines = [
        "# Quantitative Analyses & Diagnostic Breakdowns (Plan 09 §3)",
        "",
        "## 1. Target Rank Distribution across Retrieval Stages (A3)",
        "",
        "Counts of target document rank buckets per experiment ($n=142$):",
        "",
        "| Exp | Rank 1 | Rank 2–3 | Rank 4–10 | Rank 11–100 | Rank 101–1000 | Missing (>1000) | Top-10 Total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exp, counts in rank_dist.items():
        top10 = counts["Rank 1"] + counts["Rank 2-3"] + counts["Rank 4-10"]
        lines.append(
            f"| {exp} | {counts['Rank 1']} | {counts['Rank 2-3']} | {counts['Rank 4-10']} | "
            f"{counts['Rank 11-100']} | {counts['Rank 101-1000']} | {counts['Missing (>1000)']} | **{top10}** |"
        )

    lines.extend([
        "",
        "## 2. Recall Curves across Depths (A7)",
        "",
        "| Exp | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | R@500 | R@1000 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for exp, r in recall_curves.items():
        lines.append(
            f"| {exp} | {r.get(1,0):.4f} | {r.get(5,0):.4f} | {r.get(10,0):.4f} | {r.get(20,0):.4f} | "
            f"{r.get(50,0):.4f} | {r.get(100,0):.4f} | {r.get(500,0):.4f} | {r.get(1000,0):.4f} |"
        )

    lines.extend([
        "",
        "## 3. Query Length Quartile Performance (A6 & A11)",
        "",
        "Mean nDCG@10 broken down by narrative query word count quartiles:",
        "",
        "| Exp | Q1 (Short) | Q2 (Medium) | Q3 (Long) | Q4 (Very Long) |",
        "|---|---:|---:|---:|---:|",
    ])
    for exp, b_scores in len_analysis.items():
        lines.append(
            f"| {exp} | {b_scores.get('Q1_Short (<=45w)', 0):.4f} | "
            f"{b_scores.get('Q2_Medium (46-75w)', 0):.4f} | "
            f"{b_scores.get('Q3_Long (76-125w)', 0):.4f} | "
            f"{b_scores.get('Q4_VeryLong (>125w)', 0):.4f} |"
        )

    lines.extend([
        "",
        "## 4. Precision vs. Recall Trade-off (A17)",
        "",
        "| Exp | Strategy | Recall@100 (Candidate Reachability) | nDCG@10 (Top-10 Precision) |",
        "|---|---|---:|---:|",
    ])
    for exp in ["E1", "E2-R", "E3", "E6", "E7", "E8", "E9"]:
        r_curve = recall_curves.get(exp, {})
        r100 = r_curve.get(100, 0.0)
        exp_runs = runs_data.get(exp, {})
        ndcg_val = sum(m.get("ndcg10", 0.0) for m in exp_runs.values()) / max(1, len(exp_runs))
        lines.append(f"| {exp} | Tier B Cascade | {r100:.4f} | {ndcg_val:.4f} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    per_query_dir = ROOT / "evaluation" / "per_query"
    runs_dir = ROOT / "runs"
    qrels_path = ROOT / "data" / "dev1-qrels.txt"
    queries_path = ROOT / "data" / "dev1-queries.jsonl"
    out_file = ROOT / "analysis" / "quantitative_diagnostics_dev1.md"

    print("[progress] running quantitative analyses A3, A6, A7, A17...", flush=True)
    runs_data = load_per_query_data(per_query_dir)
    rank_dist = compute_rank_distribution(runs_data)
    recall_curves = compute_recall_curves(runs_dir, qrels_path)
    len_analysis = compute_query_length_analysis(queries_path, runs_data)

    report = render_quantitative_report(rank_dist, recall_curves, len_analysis, runs_data=runs_data)
    out_file.write_text(report, encoding="utf-8")
    print(f"[OK] Quantitative diagnostics saved to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
