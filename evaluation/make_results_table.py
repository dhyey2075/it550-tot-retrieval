#!/usr/bin/env python3
"""Render scope §20 results table from evaluation/results.csv (never hand-typed)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Preferred experiment order for the main table
EXP_ORDER = ["E1", "E2-R", "E2", "E3", "E4", "E5-R", "E5", "E6", "E7", "E8", "E9"]


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pick_latest_per_exp(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the last row for each exp_id (results.csv is append-only)."""
    by_exp: dict[str, dict[str, str]] = {}
    for row in rows:
        by_exp[row["exp_id"]] = row
    ordered = []
    seen = set()
    for exp in EXP_ORDER:
        if exp in by_exp:
            ordered.append(by_exp[exp])
            seen.add(exp)
    for exp, row in by_exp.items():
        if exp not in seen:
            ordered.append(row)
    return ordered


def render_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Exp. | Method (run_name) | corpus_tier | n | nDCG@10 | MRR | R@10 | R@100 | R@1000 |",
        "|---|---|---|--:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('exp_id','')} | {r.get('run_name','')} | {r.get('corpus_tier','')} | "
            f"{r.get('n_queries','')} | {r.get('ndcg10','')} | {r.get('mrr','')} | "
            f"{r.get('r10','')} | {r.get('r100','')} | {r.get('r1000','')} |"
        )
    lines.append("")
    lines.append(
        "_Caption note: with one relevant document per query, Success@10 = Recall@10; "
        "only Recall@10 is shown. Unrun experiments stay absent (do not invent TBD rows here — "
        "add them in the report template only)._"
    )
    return "\n".join(lines) + "\n"


def make_table(results_path: Path | str = Path("evaluation/results.csv")) -> str:
    path = Path(results_path)
    rows = pick_latest_per_exp(load_results(path))
    if not rows:
        return "# Results table\n\n_No rows in results.csv yet. Run experiments, then regenerate._\n"
    return "# Results table (generated from results.csv)\n\n" + render_markdown(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("evaluation/results.csv"))
    p.add_argument("--out", type=Path, default=Path("evaluation/results_table.md"))
    args = p.parse_args(argv)

    text = make_table(args.results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
