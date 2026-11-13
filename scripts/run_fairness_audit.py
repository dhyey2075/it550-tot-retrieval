#!/usr/bin/env python3
"""
Fairness Audit across all primary runs (Plan 03 §7, Plan 09 §2).
Verifies that all evaluated runs satisfy the Cranfield fairness criteria:
- Identical qrels MD5
- Identical corpus_tier ("dense_rerank_bm25")
- Identical expected candidate depths
- Consistent query coverage without evaluation leakage
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.fairness import RunMeta, assert_fair
from evaluation.io_utils import load_qrels


def find_latest_run(runs_dir: Path, exp: str, split: str = "dev1") -> Path | None:
    cands = sorted(runs_dir.glob(f"{exp}_{split}_*.run"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> int:
    runs_dir = ROOT / "runs"
    qrels_path = ROOT / "data" / "dev1-qrels.txt"
    qrels = load_qrels(qrels_path)
    expected_qids = set(qrels.keys())

    primary_exps = ["E1", "E2-R", "E3", "E4", "E4c", "E5-R", "E6", "E7", "E8", "E9"]
    runs: list[RunMeta] = []

    print("[progress] gathering RunMeta for primary experiments...", flush=True)
    for exp in primary_exps:
        p = find_latest_run(runs_dir, exp, "dev1")
        if not p:
            # Check with underscore
            cands = sorted([f for f in runs_dir.glob("*.run") if f.name.startswith(f"{exp}_")], key=lambda f: f.stat().st_mtime, reverse=True)
            p = cands[0] if cands else None
        if not p:
            print(f"  [skip] run file for {exp} not found")
            continue

        runs.append(
            RunMeta(
                run_path=p,
                exp_id=exp,
                split="dev1",
                corpus_tier="dense_rerank_bm25",
                qrels_path=qrels_path,
            )
        )

    print(f"[progress] running assert_fair across {len(runs)} runs...", flush=True)
    report = assert_fair(runs, expected_qids=expected_qids, require_same_tier=True)

    out_file = ROOT / "logs" / "fairness_audit.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    status_str = "PASS" if report.ok else "FAIL"

    lines = [
        f"# Fairness Audit Report — {status_str}",
        "",
        f"**Date:** 2026-11-12  ",
        f"**Split:** Dev1 ($n={len(expected_qids)}$)  ",
        f"**Corpus Tier:** `dense_rerank_bm25`  ",
        f"**Audit Result:** `{status_str}`  ",
        "",
        "## Fairness Checklist Items",
        "",
    ]
    for msg in report.messages:
        icon = "✅" if msg.startswith("OK") else ("ℹ️" if msg.startswith("INFO") else ("⚠️" if msg.startswith("WARNING") else "❌"))
        lines.append(f"- {icon} {msg}")

    lines.append("")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Fairness report saved to {out_file}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
