#!/usr/bin/env python3
"""Paired comparisons with bootstrap CIs and Holm correction (plan_v2/03 §5)."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from statistics import mean as _mean

PLANNED = [
    ("E1", "E2-R", "Q2 dense vs BM25 (Tier B: E2-R)"),
    ("E1", "E3", "Q3 hybrid vs BM25"),
    ("E2-R", "E3", "Q3 hybrid vs dense-rerank"),
    ("E1", "E4", "Q4 decomp helps BM25"),
    ("E2-R", "E5-R", "Q4 decomp helps dense-rerank"),
    ("E3", "E6", "Q4 decomp helps hybrid"),
    ("E3", "E7", "Q5 reformulation helps"),
    ("E6", "E8", "Q6 complementary (vs decomp)"),
    ("E7", "E8", "Q6 complementary (vs reform)"),
    ("E8", "E9", "Q7 reranking helps"),
    ("E1", "E9", "Overall system gain vs BM25"),
]


def _find_latest_per_query(per_query_dir: Path, exp: str) -> Path | None:
    # Match exact prefix like E1_ or E2-R_ or E8_
    cands = sorted(
        [p for p in per_query_dir.glob(f"{exp}_*.csv") if not p.name.endswith(".summary.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _load_per_query(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["qid"]] = {
                "ndcg10": float(row["ndcg10"]),
                "rr": float(row["rr"]),
                "r10": float(row["r10"]),
                "r100": float(row["r100"]),
                "r1000": float(row["r1000"]),
                "rank": float(row["rank_of_target"]) if row["rank_of_target"] else math.inf,
            }
    return out


def paired_bootstrap_ci(
    diffs: list[float],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean_diff, lo, hi) for paired differences."""
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo_i = int(math.floor((alpha / 2) * n_boot))
    hi_i = min(max(int(math.ceil((1 - alpha / 2) * n_boot)) - 1, 0), n_boot - 1)
    return _mean(diffs), means[lo_i], means[hi_i]


def paired_sign_flip_p(diffs: list[float], *, n_perm: int = 5000, seed: int = 42) -> float:
    """Two-sided paired randomization test on mean difference."""
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = abs(_mean(diffs))
    extreme = 0
    for _ in range(n_perm):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(_mean(flipped)) >= obs - 1e-15:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def holm_correct(pvalues: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        factor = m - rank
        val = min(1.0, pvalues[i] * factor)
        running = max(running, val)
        adj[i] = running
    return adj


def win_tie_loss(a: dict[str, dict], b: dict[str, dict], qids: list[str]) -> tuple[int, int, int]:
    win = tie = loss = 0
    for q in qids:
        ra, rb = a[q]["rank"], b[q]["rank"]
        if rb < ra:
            win += 1  # B better
        elif rb == ra:
            tie += 1
        else:
            loss += 1
    return win, tie, loss


def compare_pair(
    path_a: Path,
    path_b: Path,
    *,
    metric: str = "ndcg10",
) -> dict:
    a = _load_per_query(path_a)
    b = _load_per_query(path_b)
    qids = sorted(set(a) & set(b))
    if not qids:
        raise ValueError("no overlapping qids")
    diffs = [b[q][metric] - a[q][metric] for q in qids]
    md, lo, hi = paired_bootstrap_ci(diffs)
    p = paired_sign_flip_p(diffs)
    w, t, l = win_tie_loss(a, b, qids)
    return {
        "n": len(qids),
        "metric": metric,
        "mean_diff_B_minus_A": md,
        "ci_low": lo,
        "ci_high": hi,
        "p_raw": p,
        "wins_B": w,
        "ties": t,
        "losses_B": l,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-query-dir", type=Path, default=Path("evaluation/per_query"))
    p.add_argument("--a", type=Path, help="per-query CSV for run A")
    p.add_argument("--b", type=Path, help="per-query CSV for run B")
    p.add_argument("--metric", default="ndcg10", choices=["ndcg10", "rr", "r10", "r100", "r1000"])
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", type=Path, default=Path("analysis/planned_comparisons.csv"))
    args = p.parse_args(argv)

    if args.a and args.b:
        path_a = args.a if args.a.exists() else (args.per_query_dir / f"{args.a}.csv" if (args.per_query_dir / f"{args.a}.csv").exists() else args.per_query_dir / args.a)
        path_b = args.b if args.b.exists() else (args.per_query_dir / f"{args.b}.csv" if (args.per_query_dir / f"{args.b}.csv").exists() else args.per_query_dir / args.b)
        res = compare_pair(path_a, path_b, metric=args.metric)
        print(f"{args.label_a} vs {args.label_b} ({args.metric})")
        for k, v in res.items():
            print(f"  {k}: {v}")
        return 0

    print(f"Running all planned comparisons on {args.per_query_dir} (metric: {args.metric})...")
    rows = []
    for exp_a, exp_b, desc in PLANNED:
        file_a = _find_latest_per_query(args.per_query_dir, exp_a)
        file_b = _find_latest_per_query(args.per_query_dir, exp_b)
        if not file_a or not file_b:
            print(f"  [skip] {exp_a} vs {exp_b} (missing: a={bool(file_a)}, b={bool(file_b)})")
            continue
        res = compare_pair(file_a, file_b, metric=args.metric)
        res["exp_a"] = exp_a
        res["exp_b"] = exp_b
        res["description"] = desc
        rows.append(res)

    if not rows:
        print("No paired comparisons could be computed.")
        return 0

    p_raws = [r["p_raw"] for r in rows]
    p_holms = holm_correct(p_raws)
    for r, ph in zip(rows, p_holms):
        r["p_holm"] = ph

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "description", "exp_a", "exp_b", "metric", "n",
        "mean_diff_B_minus_A", "ci_low", "ci_high", "p_raw", "p_holm",
        "wins_B", "ties", "losses_B",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"[OK] Planned comparisons written to {args.out}")

    print("\n| Comparison | Exp A vs B | Metric | n | Mean Diff (B - A) | 95% CI | p (raw) | p (Holm) | W / T / L |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        wtl_str = f"{r['wins_B']} / {r['ties']} / {r['losses_B']}"
        print(f"| {r['description']} | {r['exp_a']} vs {r['exp_b']} | {r['metric']} | {r['n']} | {r['mean_diff_B_minus_A']:+.4f} | {ci_str} | {r['p_raw']:.4f} | {r['p_holm']:.4f} | {wtl_str} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
