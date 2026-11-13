"""
Combined decomposition + reformulation fusion (plan_v2/07, experiment E8).

Implements fusion protocols:
- F1: Flat RRF over all facet lists + reformulation lists
- F2: Two-level branch-wise RRF (Branch D [Decomp] + Branch R [Reformulation])
- F3: Weighted branch-wise RRF
- F4: Paper-faithful candidate (F1 or F2 plus original query lists)
Also exports candidate pool for cross-encoder reranking (Plan 08).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from evaluation.io_utils import write_trec_run
from retrieval.bm25.search import Bm25Backend
from retrieval.dense.rerank import DenseReranker
from retrieval.queries import load_queries
from retrieval.rrf import rrf_fuse, rrf_fuse_docids
from tot.decomposition import facet_queries

DocTextFn = Callable[[str], str]


def _sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps({**payload, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )


def weighted_rrf_fuse(
    lists_with_weights: list[tuple[Sequence[str], float]],
    *,
    k: int = 60,
    out_depth: int = 1000,
) -> list[str]:
    """Weighted RRF over a sequence of (docid_list, weight) tuples."""
    scores: dict[str, float] = defaultdict(float)
    for docids, weight in lists_with_weights:
        for rank, docid in enumerate(docids, start=1):
            scores[str(docid)] += weight * (1.0 / (k + rank))
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [docid for docid, _ in ordered[:out_depth]]


def run_e8(
    queries: list[dict[str, str]],
    facets_by_qid: dict[str, dict[str, str]],
    reformulations_by_qid: dict[str, str],
    bm25: Bm25Backend | None = None,
    reranker: DenseReranker | None = None,
    doc_text_fn: DocTextFn | None = None,
    *,
    out_path: str | Path,
    run_name: str,
    protocol: str = "f2",  # f1 | f2 | f3 | f4
    weight_d: float = 1.0,
    weight_r: float = 1.0,
    include_original: bool = False,
    branch_d_rankings: dict[str, list[str]] | None = None,
    branch_r_rankings: dict[str, list[str]] | None = None,
    branch_orig_rankings: dict[str, list[str]] | None = None,
    rrf_k: int = 60,
    depth: int = 1000,
    config: dict[str, Any] | None = None,
) -> Path:
    """
    Execute E8 combined fusion pipeline across queries.
    Uses precomputed branch rankings if provided (fast mode, plan 07 §2),
    or executes live retrieval per facet/rewrite if not.
    """
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}

    for row in queries:
        qid, orig_query = row["qid"], row["query"]
        facets = facets_by_qid.get(qid) or {}
        fqueries = facet_queries(facets, fallback_query=orig_query)
        reform_query = reformulations_by_qid.get(qid, orig_query)

        # 1. Decomposition branch (from precomputed or live retrieval)
        decomp_bm25_lists: list[list[str]] = []
        decomp_dense_lists: list[list[str]] = []
        if branch_d_rankings and qid in branch_d_rankings:
            branch_d = branch_d_rankings[qid][:depth]
        else:
            if bm25 is None or reranker is None or doc_text_fn is None:
                raise ValueError("bm25, reranker, and doc_text_fn required when branch_d_rankings not supplied")
            for fq in fqueries:
                b_hits = [d for d, _ in bm25.search(fq, k=depth)]
                d_hits = [d for d, _ in reranker.rerank(fq, b_hits, doc_text_fn, top_k=depth)]
                decomp_bm25_lists.append(b_hits)
                decomp_dense_lists.append(d_hits)
            branch_d = rrf_fuse_docids(decomp_bm25_lists + decomp_dense_lists, k=rrf_k, out_depth=depth)

        # 2. Reformulation branch
        ref_b_hits: list[str] = []
        ref_d_hits: list[str] = []
        if branch_r_rankings and qid in branch_r_rankings:
            branch_r = branch_r_rankings[qid][:depth]
        else:
            if bm25 is None or reranker is None or doc_text_fn is None:
                raise ValueError("bm25, reranker, and doc_text_fn required when branch_r_rankings not supplied")
            ref_b_hits = [d for d, _ in bm25.search(reform_query, k=depth)]
            ref_d_hits = [d for d, _ in reranker.rerank(reform_query, ref_b_hits, doc_text_fn, top_k=depth)]
            branch_r = rrf_fuse_docids([ref_b_hits, ref_d_hits], k=rrf_k, out_depth=depth)

        # 3. Optional Original branch
        orig_b_hits: list[str] = []
        orig_d_hits: list[str] = []
        branch_orig: list[str] = []
        if include_original:
            if branch_orig_rankings and qid in branch_orig_rankings:
                branch_orig = branch_orig_rankings[qid][:depth]
            else:
                if bm25 is None or reranker is None or doc_text_fn is None:
                    raise ValueError("bm25, reranker, and doc_text_fn required when branch_orig_rankings not supplied")
                orig_b_hits = [d for d, _ in bm25.search(orig_query, k=depth)]
                orig_d_hits = [d for d, _ in reranker.rerank(orig_query, orig_b_hits, doc_text_fn, top_k=depth)]
                branch_orig = rrf_fuse_docids([orig_b_hits, orig_d_hits], k=rrf_k, out_depth=depth)

        # 4. Combine according to chosen protocol
        prot = protocol.lower()
        if prot == "f1":
            # Flat RRF over all sub-lists if available, or branch-level flat RRF
            if decomp_bm25_lists and ref_b_hits:
                all_lists = decomp_bm25_lists + decomp_dense_lists + [ref_b_hits, ref_d_hits]
                if include_original:
                    all_lists.extend([orig_b_hits, orig_d_hits])
                fused = rrf_fuse_docids(all_lists, k=rrf_k, out_depth=depth)
            else:
                branches = [branch_d, branch_r]
                if include_original:
                    branches.append(branch_orig)
                fused = rrf_fuse_docids(branches, k=rrf_k, out_depth=depth)
        elif prot == "f2":
            # Two-level branch-wise RRF: 50% Decomp + 50% Reform
            branches = [branch_d, branch_r]
            if include_original:
                branches.append(branch_orig)
            fused = rrf_fuse_docids(branches, k=rrf_k, out_depth=depth)
        elif prot == "f3":
            # Weighted branch-wise RRF
            weighted_branches = [(branch_d, weight_d), (branch_r, weight_r)]
            if include_original:
                weighted_branches.append((branch_orig, 1.0))
            fused = weighted_rrf_fuse(weighted_branches, k=rrf_k, out_depth=depth)
        elif prot == "f4":
            # Paper-faithful: branch D + branch R + base original
            branches = [branch_d, branch_r, branch_orig]
            fused = rrf_fuse_docids(branches, k=rrf_k, out_depth=depth)
        else:
            raise ValueError(f"Unknown fusion protocol {protocol}; choose from f1, f2, f3, f4")

        rankings[qid] = fused
        scores[qid] = [float(len(fused) - i) for i in range(len(fused))]

    write_trec_run(out_path, rankings, run_name=run_name, scores=scores)
    _sidecar(
        out_path,
        {
            "exp": "E8",
            "run_name": run_name,
            "protocol": protocol,
            "weight_d": weight_d,
            "weight_r": weight_r,
            "include_original": include_original,
            "rrf_k": rrf_k,
            "config": config or {},
        },
    )
    return out_path


def run_e8_suite(
    query_path: str | Path,
    facets_by_qid: dict[str, dict[str, str]],
    reformulations_by_qid: dict[str, str],
    bm25: Bm25Backend | None = None,
    reranker: DenseReranker | None = None,
    doc_text_fn: DocTextFn | None = None,
    *,
    runs_dir: str | Path,
    split: str,
    e6_run_path: str | Path | None = None,
    e7_run_path: str | Path | None = None,
    e3_run_path: str | Path | None = None,
    rrf_k: int = 60,
    depth: int = 1000,
    date: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Execute F1, F2, and F4 protocols for Dev1 comparison (D-08). Reuses cached runs if present."""
    from evaluation.io_utils import load_run
    from tot.run_naming import make_run_name

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
    queries = load_queries(query_path)
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    def _find_run(prefix: str) -> Path | None:
        matches = sorted(runs_dir.glob(f"{prefix}_{split}_*.run"), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None

    if e6_run_path is None:
        e6_run_path = _find_run("E6")
    if e7_run_path is None:
        e7_run_path = _find_run("E7")
    if e3_run_path is None:
        e3_run_path = _find_run("E3")

    branch_d = load_run(e6_run_path) if e6_run_path and Path(e6_run_path).exists() else None
    branch_r = load_run(e7_run_path) if e7_run_path and Path(e7_run_path).exists() else None
    branch_orig = load_run(e3_run_path) if e3_run_path and Path(e3_run_path).exists() else None

    results: dict[str, Path] = {}

    # Main E8 protocol (F2: two-level branch-wise)
    e8_f2_name = make_run_name("E8", split, "tierB", "f2_twolevel", date)
    e8_f2_path = runs_dir / f"{e8_f2_name}.run"
    run_e8(
        queries,
        facets_by_qid,
        reformulations_by_qid,
        bm25,
        reranker,
        doc_text_fn,
        out_path=e8_f2_path,
        run_name=e8_f2_name,
        protocol="f2",
        branch_d_rankings=branch_d,
        branch_r_rankings=branch_r,
        branch_orig_rankings=branch_orig,
        rrf_k=rrf_k,
        depth=depth,
        config=config,
    )
    results["E8"] = e8_f2_path
    results["E8-F2"] = e8_f2_path

    # Protocol F1 (flat RRF)
    e8_f1_name = make_run_name("E8", split, "tierB", "f1_flat", date)
    e8_f1_path = runs_dir / f"{e8_f1_name}.run"
    run_e8(
        queries,
        facets_by_qid,
        reformulations_by_qid,
        bm25,
        reranker,
        doc_text_fn,
        out_path=e8_f1_path,
        run_name=e8_f1_name,
        protocol="f1",
        branch_d_rankings=branch_d,
        branch_r_rankings=branch_r,
        branch_orig_rankings=branch_orig,
        rrf_k=rrf_k,
        depth=depth,
        config=config,
    )
    results["E8-F1"] = e8_f1_path

    # Protocol F4 (with original query)
    e8_f4_name = make_run_name("E8", split, "tierB", "f4_plusorig", date)
    e8_f4_path = runs_dir / f"{e8_f4_name}.run"
    run_e8(
        queries,
        facets_by_qid,
        reformulations_by_qid,
        bm25,
        reranker,
        doc_text_fn,
        out_path=e8_f4_path,
        run_name=e8_f4_name,
        protocol="f4",
        include_original=True,
        branch_d_rankings=branch_d,
        branch_r_rankings=branch_r,
        branch_orig_rankings=branch_orig,
        rrf_k=rrf_k,
        depth=depth,
        config=config,
    )
    results["E8-F4"] = e8_f4_path

    return results


def save_candidate_pool_for_reranking(
    run_path: str | Path,
    doc_text_fn: DocTextFn,
    out_jsonl: str | Path,
    *,
    top_k: int = 100,
) -> Path:
    """Save top-K candidates per query with document text for Plan 08 cross-encoder."""
    from evaluation.io_utils import load_run

    rankings = load_run(run_path)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for qid, docids in rankings.items():
            pool = []
            for rank, docid in enumerate(docids[:top_k], start=1):
                text = doc_text_fn(str(docid))
                pool.append({"rank": rank, "docid": str(docid), "text": text})
            f.write(json.dumps({"qid": qid, "candidates": pool}, ensure_ascii=False) + "\n")
    return out_jsonl
