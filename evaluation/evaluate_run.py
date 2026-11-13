#!/usr/bin/env python3
"""Evaluate one TREC run against a qrels file and append to results.csv."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.io_utils import md5_file
from evaluation.metrics import bootstrap_ci, evaluate_run

RESULTS_HEADER = [
    "exp_id",
    "run_name",
    "split",
    "corpus_tier",
    "n_queries",
    "ndcg10",
    "mrr",
    "r10",
    "r100",
    "r1000",
    "map10",
    "ci_low_ndcg10",
    "ci_high_ndcg10",
    "ci_low_mrr",
    "ci_high_mrr",
    "n_missing_from_run",
    "not_found_rate",
    "qrels_md5",
    "config_hash",
    "date",
]


def _append_results(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULTS_HEADER})


def _write_per_query(path: Path, per_query) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["qid", "target_docid", "rank_of_target", "rr", "ndcg10", "r10", "r100", "r1000", "map10"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pq in per_query:
            w.writerow(pq.to_row())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a TREC run (plan_v2/03)")
    p.add_argument("run", type=Path, help="Path to .run file")
    p.add_argument("--exp", required=True, help="Experiment id, e.g. E1 or E2-R")
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--split", default="dev3")
    p.add_argument("--corpus-tier", default="dense_rerank_bm25")
    p.add_argument("--config-hash", default="")
    p.add_argument("--results", type=Path, default=Path("evaluation/results.csv"))
    p.add_argument("--per-query-dir", type=Path, default=Path("evaluation/per_query"))
    p.add_argument("--no-append", action="store_true", help="Print only; do not write results.csv")
    p.add_argument("--allow-test-prefix", action="store_true", help="Allow TEST_ run names into results (discouraged)")
    args = p.parse_args(argv)

    run_name = args.run.stem
    if run_name.startswith("TEST_") and not args.no_append and not args.allow_test_prefix:
        raise SystemExit(
            f"Refusing to append {run_name} to results.csv (TEST_ prefix). "
            "Use --no-append or --allow-test-prefix."
        )

    per_q, agg = evaluate_run(args.run, args.qrels)
    ci_ndcg = bootstrap_ci([pq.ndcg10 for pq in per_q])
    ci_mrr = bootstrap_ci([pq.rr for pq in per_q])

    print(f"run={run_name} exp={args.exp} split={args.split} tier={args.corpus_tier}")
    print(
        f"n={agg.n_queries} missing_from_run={agg.n_missing_from_run} "
        f"ndcg10={agg.ndcg10:.6f} mrr={agg.mrr:.6f} "
        f"r10={agg.r10:.6f} r100={agg.r100:.6f} r1000={agg.r1000:.6f}"
    )
    print(f"ndcg10 95% CI [{ci_ndcg[0]:.6f}, {ci_ndcg[1]:.6f}]")
    print(f"mrr    95% CI [{ci_mrr[0]:.6f}, {ci_mrr[1]:.6f}]")
    print(f"not_found_rate={agg.not_found_rate:.6f}")

    per_path = args.per_query_dir / f"{run_name}.csv"
    _write_per_query(per_path, per_q)
    print(f"wrote {per_path}")

    if not args.no_append:
        row = {
            "exp_id": args.exp,
            "run_name": run_name,
            "split": args.split,
            "corpus_tier": args.corpus_tier,
            "n_queries": agg.n_queries,
            "ndcg10": f"{agg.ndcg10:.6f}",
            "mrr": f"{agg.mrr:.6f}",
            "r10": f"{agg.r10:.6f}",
            "r100": f"{agg.r100:.6f}",
            "r1000": f"{agg.r1000:.6f}",
            "map10": f"{agg.map10:.6f}",
            "ci_low_ndcg10": f"{ci_ndcg[0]:.6f}",
            "ci_high_ndcg10": f"{ci_ndcg[1]:.6f}",
            "ci_low_mrr": f"{ci_mrr[0]:.6f}",
            "ci_high_mrr": f"{ci_mrr[1]:.6f}",
            "n_missing_from_run": agg.n_missing_from_run,
            "not_found_rate": f"{agg.not_found_rate:.6f}",
            "qrels_md5": md5_file(args.qrels),
            "config_hash": args.config_hash,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        _append_results(args.results, row)
        print(f"appended {args.results}")

    # sidecar summary next to per-query for debugging
    summary_path = args.per_query_dir / f"{run_name}.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "exp_id": args.exp,
                "aggregate": agg.to_row(),
                "ci_ndcg10": ci_ndcg,
                "ci_mrr": ci_mrr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
