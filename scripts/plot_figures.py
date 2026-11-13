#!/usr/bin/env python3
"""
Generate publication-quality figures for Plan 09 (Scope §18-§20):
- fig_a3_rank_distribution.png: Stacked bar chart of rank-of-target buckets
- fig_a7_recall_curves.png: Recall@K curves
- fig_a8_complementarity.png: Branch complementarity & oracle union
- fig_a17_precision_recall.png: Precision vs Recall trade-off trajectory
- fig_error_taxonomy.png: Failure cause distribution
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure matplotlib uses headless backend and writable temp dir
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_tot")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.quantitative_analyses import (
    EXP_ORDER,
    compute_rank_distribution,
    compute_recall_curves,
    load_per_query_data,
)


def plot_rank_distribution(rank_dist: dict[str, dict[str, int]], out_path: Path) -> None:
    """Stacked bar chart of rank-of-target buckets (A3)."""
    exps = [e for e in ["E1", "E2-R", "E3", "E4", "E4c", "E5-R", "E6", "E7", "E8", "E9"] if e in rank_dist]
    buckets = ["Rank 1", "Rank 2-3", "Rank 4-10", "Rank 11-100", "Rank 101-1000", "Missing (>1000)"]
    colors = ["#1b7837", "#7fbf7b", "#a6dba0", "#d9f0d3", "#f7f7f7", "#e0e0e0"]

    data = np.array([[rank_dist[e][b] for b in buckets] for e in exps])

    plt.figure(figsize=(10, 5), dpi=300)
    bottom = np.zeros(len(exps))
    for i, b in enumerate(buckets):
        plt.bar(exps, data[:, i], bottom=bottom, label=b, color=colors[i], edgecolor="gray", linewidth=0.5)
        bottom += data[:, i]

    plt.title("Rank-of-Target Document Distribution across Pipeline Phases (Dev1, n=142)", fontsize=12, fontweight="bold")
    plt.xlabel("Experiment / Retrieval Pipeline Phase", fontsize=10)
    plt.ylabel("Number of Queries", fontsize=10)
    plt.ylim(0, 142)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path.name}")


def plot_recall_curves(recall_curves: dict[str, dict[int, float]], out_path: Path) -> None:
    """Recall@K curve comparison (A7)."""
    cutoffs = [1, 5, 10, 20, 50, 100, 500, 1000]
    palette = {
        "E1": ("#4d4d4d", "-", "o"),
        "E2-R": ("#2166ac", "--", "s"),
        "E3": ("#4393c3", "-.", "^"),
        "E4c": ("#d6604d", ":", "v"),
        "E6": ("#b2182b", "-", "d"),
        "E7": ("#f4a582", "--", ">"),
        "E8": ("#762a83", "-.", "<"),
        "E9": ("#00441b", "-", "P"),
    }

    plt.figure(figsize=(9, 5), dpi=300)
    for exp, (color, ls, marker) in palette.items():
        if exp not in recall_curves:
            continue
        ys = [recall_curves[exp].get(k, 0.0) for k in cutoffs]
        plt.plot(cutoffs, ys, label=f"{exp}", color=color, linestyle=ls, marker=marker, markersize=5, linewidth=1.5)

    plt.xscale("log")
    plt.xticks(cutoffs, [str(k) for k in cutoffs])
    plt.title("Candidate Pool Reachability: Recall@K Curves (Dev1, Tier B)", fontsize=12, fontweight="bold")
    plt.xlabel("Retrieval Cutoff Depth K (log scale)", fontsize=10)
    plt.ylabel("Recall@K (Fraction of Target Retrieved)", fontsize=10)
    plt.ylim(0, 0.30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path.name}")


def plot_precision_recall_tradeoff(rank_dist: dict[str, dict[str, int]], recall_curves: dict[str, dict[int, float]], runs_data: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    """Precision (nDCG@10) vs Candidate Reachability (Recall@100) (A17)."""
    exps = ["E1", "E2-R", "E3", "E4c", "E6", "E7", "E8", "E9"]

    plt.figure(figsize=(8, 5), dpi=300)
    for exp in exps:
        r100 = recall_curves.get(exp, {}).get(100, 0.0)
        exp_runs = runs_data.get(exp, {})
        ndcg = sum(m.get("ndcg10", 0.0) for m in exp_runs.values()) / max(1, len(exp_runs))

        color = "#1b7837" if exp in ["E8", "E9"] else ("#2166ac" if "E2" in exp or "E3" in exp else "#b2182b")
        plt.scatter(r100, ndcg, s=120, color=color, zorder=5)
        offset_y = 0.001 if exp != "E3" else -0.002
        plt.annotate(exp, (r100, ndcg), textcoords="offset points", xytext=(8, offset_y * 1000), fontweight="bold")

    plt.title("Recall-vs-Precision Trade-off: Recall@100 vs. nDCG@10 (Dev1)", fontsize=12, fontweight="bold")
    plt.xlabel("Candidate Pool Reachability (Recall@100)", fontsize=10)
    plt.ylabel("Top-10 Precision (nDCG@10)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path.name}")


def plot_error_taxonomy(sheet_path: Path, out_path: Path) -> None:
    """Horizontal bar chart of error taxonomy causes."""
    if not sheet_path.exists():
        return
    counts: dict[str, int] = {}
    with sheet_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            lbl = r.get("primary_label", "Unknown")
            counts[lbl] = counts.get(lbl, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: x[1])
    labels = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]

    plt.figure(figsize=(8, 4.5), dpi=300)
    bars = plt.barh(labels, vals, color="#386cb0", edgecolor="navy", height=0.6)
    plt.title("Failure Taxonomy Distribution (Annotated Error Sample, n=94)", fontsize=11, fontweight="bold")
    plt.xlabel("Number of Queries", fontsize=10)
    plt.xlim(0, max(vals) + 5)
    for b in bars:
        w = b.get_width()
        plt.text(w + 0.8, b.get_y() + b.get_height() / 2, f"{w} ({w/sum(vals)*100:.1f}%)", va="center", fontsize=8)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path.name}")


def main() -> int:
    fig_dir = ROOT / "analysis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_query_dir = ROOT / "evaluation" / "per_query"
    runs_dir = ROOT / "runs"
    qrels_path = ROOT / "data" / "dev1-qrels.txt"
    sheet_path = ROOT / "analysis" / "error_sheet.csv"

    print("[progress] generating diagnostic figures in analysis/figures/...", flush=True)
    r_data = load_per_query_data(per_query_dir)
    r_dist = compute_rank_distribution(r_data)
    r_curves = compute_recall_curves(runs_dir, qrels_path)

    plot_rank_distribution(r_dist, fig_dir / "fig_a3_rank_distribution.png")
    plot_recall_curves(r_curves, fig_dir / "fig_a7_recall_curves.png")
    plot_precision_recall_tradeoff(r_dist, r_curves, r_data, fig_dir / "fig_a17_precision_recall.png")
    plot_error_taxonomy(sheet_path, fig_dir / "fig_error_taxonomy.png")

    print(f"[OK] All figures generated in {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
