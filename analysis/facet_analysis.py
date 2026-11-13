#!/usr/bin/env python3
"""
Per-facet expert retrieval analysis (Plan 09 §3, Analysis A13).
Measures Recall@5, 10, 20, 100 for each facet alone, and non-N/A rate (analogue of Lin et al. Table 6).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.io_utils import load_qrels
from retrieval.bm25.search import PyseriniSearcher
from retrieval.pipeline_e4e6 import load_cached_facets
from retrieval.queries import load_queries
from tot.decomposition import FACET_KEYS, non_na_rates


def find_index(parent: Path) -> Path:
    marker = parent / "INDEX_READY"
    if marker.exists():
        p = Path(marker.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    cands = [p for p in parent.rglob("*") if p.is_dir() and any(p.glob("segments*"))]
    if not cands:
        raise FileNotFoundError(f"No Lucene index under {parent}")
    return max(cands, key=lambda p: sum(1 for _ in p.glob("*")))


def evaluate_facet_retrieval(
    queries: list[dict[str, str]],
    facets_by_qid: dict[str, dict[str, str]],
    qrels: dict[str, str],
    bm25: PyseriniSearcher,
    cutoffs: tuple[int, ...] = (5, 10, 20, 100),
) -> dict[str, dict[str, float]]:
    """Evaluate each facet's isolated BM25 recall on active queries."""
    total_q = len(queries)
    results: dict[str, dict[str, float]] = {}

    for facet in FACET_KEYS:
        active_q = 0
        hits_at_k = {k: 0 for k in cutoffs}

        for row in queries:
            qid = row["qid"]
            target = qrels.get(qid)
            if not target:
                continue
            val = facets_by_qid.get(qid, {}).get(facet, "N/A")
            if val == "N/A" or not val.strip():
                continue

            active_q += 1
            search_hits = bm25.search(val, k=max(cutoffs))
            retrieved_ids = [d for d, _ in search_hits]

            for k in cutoffs:
                if target in retrieved_ids[:k]:
                    hits_at_k[k] += 1

        recalls = {
            f"R@{k}": (hits_at_k[k] / total_q) for k in cutoffs
        }
        recalls["active_rate"] = active_q / total_q
        recalls["active_count"] = active_q
        results[facet] = recalls

    return results


def render_facet_table(results: dict[str, dict[str, float]], rates: dict[str, float]) -> str:
    lines = [
        "# Per-Facet Expert Retrieval Analysis (Plan 09 §3, Analysis A13)",
        "",
        "Analogue of Lin et al. (2023) Table 6, evaluating the isolated retrieval power of each facet.",
        "",
        "| Facet Field | Non-N/A Rate | Active Queries | R@5 | R@10 | R@20 | R@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for facet in FACET_KEYS:
        r = results.get(facet, {})
        rate = rates.get(facet, r.get("active_rate", 0.0)) * 100
        cnt = int(r.get("active_count", 0))
        lines.append(
            f"| **{facet}** | {rate:.1f}% | {cnt} | "
            f"{r.get('R@5', 0.0):.4f} | {r.get('R@10', 0.0):.4f} | "
            f"{r.get('R@20', 0.0):.4f} | {r.get('R@100', 0.0):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    split = "dev1"
    qrels_path = ROOT / "data" / f"{split}-qrels.txt"
    queries_path = ROOT / "data" / f"{split}-queries.jsonl"
    facets_path = ROOT / "cache" / f"facets_{split}_extractive.jsonl"
    index_dir = find_index(ROOT / "indices" / "anserini")

    print("[progress] loading queries and facets for facet expert analysis...", flush=True)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    facets = load_cached_facets(facets_path)

    # Compute non-N/A rates
    facet_rows = [facets.get(q["qid"], {}) for q in queries]
    rates = non_na_rates(facet_rows)

    bm25 = PyseriniSearcher(index_dir)
    print(f"[progress] evaluating per-facet BM25 retrieval across {len(queries)} queries...", flush=True)
    results = evaluate_facet_retrieval(queries, facets, qrels, bm25)

    rep = render_facet_table(results, rates)
    out_file = ROOT / "analysis" / f"facet_expert_table_{split}.md"
    out_file.write_text(rep, encoding="utf-8")
    print(f"[OK] Per-facet expert table saved to {out_file}")
    print("\n" + rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
