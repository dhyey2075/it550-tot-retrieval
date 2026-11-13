"""
Query reformulation retrieval pipeline: E7 and diagnostics (plan_v2/06).

E7: RRF(BM25(reformulated), Dense(reformulated)) under Tier B
E7-bm25: BM25(reformulated) only
E7-dense: Dense(reformulated) only (reranked from BM25(reformulated) candidates)
E7-split: RRF(BM25(reform_keyword), Dense(reform_descriptive))
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation.io_utils import write_trec_run
from retrieval.bm25.search import Bm25Backend
from retrieval.dense.rerank import DenseReranker
from retrieval.queries import load_queries
from retrieval.rrf import rrf_fuse_docids

DocTextFn = Callable[[str], str]


def _sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps({**payload, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )


def run_e7_bm25(
    queries: list[dict[str, str]],
    reformulations_by_qid: dict[str, str],
    bm25: Bm25Backend,
    *,
    out_path: str | Path,
    run_name: str,
    depth: int = 1000,
    config: dict[str, Any] | None = None,
) -> Path:
    """Diagnostic: BM25 over reformulated queries alone."""
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    fallbacks = 0

    for row in queries:
        qid, orig_query = row["qid"], row["query"]
        q_text = reformulations_by_qid.get(qid, orig_query)
        if q_text == orig_query:
            fallbacks += 1
        hits = bm25.search(q_text, k=depth)
        rankings[qid] = [d for d, _ in hits]
        scores[qid] = [s for _, s in hits]

    write_trec_run(out_path, rankings, run_name=run_name, scores=scores)
    _sidecar(out_path, {"exp": "E7-bm25", "run_name": run_name, "fallbacks": fallbacks, "config": config or {}})
    return out_path


def run_e7_dense(
    queries: list[dict[str, str]],
    reformulations_by_qid: dict[str, str],
    bm25_rankings: dict[str, list[str]],
    reranker: DenseReranker,
    doc_text_fn: DocTextFn,
    *,
    out_path: str | Path,
    run_name: str,
    top_k: int = 1000,
    config: dict[str, Any] | None = None,
) -> Path:
    """Diagnostic: Dense reranking of BM25(reformulated) candidates using reformulated query."""
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}

    for row in queries:
        qid, orig_query = row["qid"], row["query"]
        q_text = reformulations_by_qid.get(qid, orig_query)
        cands = bm25_rankings.get(qid, [])[:top_k]
        hits = reranker.rerank(q_text, cands, doc_text_fn, top_k=top_k)
        rankings[qid] = [d for d, _ in hits]
        scores[qid] = [s for _, s in hits]

    write_trec_run(out_path, rankings, run_name=run_name, scores=scores)
    _sidecar(out_path, {"exp": "E7-dense", "run_name": run_name, "config": config or {}})
    return out_path


def run_e7(
    queries: list[dict[str, str]],
    reformulations_by_qid: dict[str, str],
    bm25: Bm25Backend,
    reranker: DenseReranker,
    doc_text_fn: DocTextFn,
    *,
    out_path: str | Path,
    run_name: str,
    rrf_k: int = 60,
    depth: int = 1000,
    config: dict[str, Any] | None = None,
    out_bm25_path: str | Path | None = None,
    bm25_run_name: str | None = None,
    out_dense_path: str | Path | None = None,
    dense_run_name: str | None = None,
) -> Path:
    """Full E7: RRF over BM25(reformulated) and Dense(reformulated) lists."""
    out_path = Path(out_path)
    bm25_rankings: dict[str, list[str]] = {}
    dense_rankings: dict[str, list[str]] = {}
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    fallbacks = 0

    for row in queries:
        qid, orig_query = row["qid"], row["query"]
        q_text = reformulations_by_qid.get(qid, orig_query)
        if q_text == orig_query:
            fallbacks += 1

        # BM25 on rewrite
        hits = bm25.search(q_text, k=depth)
        b_list = [d for d, _ in hits]
        bm25_rankings[qid] = b_list

        # Dense rerank on rewrite
        dense_hits = reranker.rerank(q_text, b_list[:depth], doc_text_fn, top_k=depth)
        d_list = [d for d, _ in dense_hits]
        dense_rankings[qid] = d_list

        # RRF fusion
        fused = rrf_fuse_docids([b_list, d_list], k=rrf_k, out_depth=depth)
        rankings[qid] = fused
        scores[qid] = [float(len(fused) - i) for i in range(len(fused))]

    write_trec_run(out_path, rankings, run_name=run_name, scores=scores)
    _sidecar(
        out_path,
        {
            "exp": "E7",
            "run_name": run_name,
            "fallbacks": fallbacks,
            "rrf_k": rrf_k,
            "config": config or {},
        },
    )

    if out_bm25_path:
        out_bm25_path = Path(out_bm25_path)
        write_trec_run(out_bm25_path, bm25_rankings, run_name=bm25_run_name or run_name)
        _sidecar(out_bm25_path, {"exp": "E7-bm25", "run_name": bm25_run_name or run_name, "config": config or {}})

    if out_dense_path:
        out_dense_path = Path(out_dense_path)
        write_trec_run(out_dense_path, dense_rankings, run_name=dense_run_name or run_name)
        _sidecar(out_dense_path, {"exp": "E7-dense", "run_name": dense_run_name or run_name, "config": config or {}})

    return out_path


def run_e7_split(
    queries: list[dict[str, str]],
    keyword_reformulations: dict[str, str],
    descriptive_reformulations: dict[str, str],
    bm25: Bm25Backend,
    reranker: DenseReranker,
    doc_text_fn: DocTextFn,
    *,
    out_path: str | Path,
    run_name: str,
    rrf_k: int = 60,
    depth: int = 1000,
    config: dict[str, Any] | None = None,
) -> Path:
    """Diagnostic extension: R-K (keyword) -> BM25, R-N (descriptive) -> Dense rerank."""
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}

    for row in queries:
        qid, orig_query = row["qid"], row["query"]
        q_kw = keyword_reformulations.get(qid, orig_query)
        q_desc = descriptive_reformulations.get(qid, orig_query)

        hits = bm25.search(q_kw, k=depth)
        b_list = [d for d, _ in hits]

        dense_hits = reranker.rerank(q_desc, b_list[:depth], doc_text_fn, top_k=depth)
        d_list = [d for d, _ in dense_hits]

        fused = rrf_fuse_docids([b_list, d_list], k=rrf_k, out_depth=depth)
        rankings[qid] = fused
        scores[qid] = [float(len(fused) - i) for i in range(len(fused))]

    write_trec_run(out_path, rankings, run_name=run_name, scores=scores)
    _sidecar(out_path, {"exp": "E7-split", "run_name": run_name, "rrf_k": rrf_k, "config": config or {}})
    return out_path


def run_e7_suite(
    query_path: str | Path,
    reformulations_by_qid: dict[str, str],
    bm25: Bm25Backend,
    reranker: DenseReranker,
    doc_text_fn: DocTextFn,
    *,
    runs_dir: str | Path,
    split: str,
    style: str = "keyword",
    descriptive_by_qid: dict[str, str] | None = None,
    rrf_k: int = 60,
    depth: int = 1000,
    date: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    from tot.run_naming import make_run_name

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
    queries = load_queries(query_path)
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main E7 (computes BM25 and Dense on rewrite, fusing with RRF and outputting diagnostics)
    e7_name = make_run_name("E7", split, "tierB", style, date)
    e7_path = runs_dir / f"{e7_name}.run"
    e7_bm25_name = make_run_name("E7-bm25", split, "tierB", style, date)
    e7_bm25_path = runs_dir / f"{e7_bm25_name}.run"
    e7_dense_name = make_run_name("E7-dense", split, "tierB", style, date)
    e7_dense_path = runs_dir / f"{e7_dense_name}.run"

    run_e7(
        queries,
        reformulations_by_qid,
        bm25,
        reranker,
        doc_text_fn,
        out_path=e7_path,
        run_name=e7_name,
        rrf_k=rrf_k,
        depth=depth,
        config=config,
        out_bm25_path=e7_bm25_path,
        bm25_run_name=e7_bm25_name,
        out_dense_path=e7_dense_path,
        dense_run_name=e7_dense_name,
    )

    out = {
        "E7": e7_path,
        "E7-bm25": e7_bm25_path,
        "E7-dense": e7_dense_path,
    }

    # 4. Optional E7-split if descriptive reformulations provided
    if descriptive_by_qid:
        e7_split_name = make_run_name("E7-split", split, "tierB", "split", date)
        e7_split_path = runs_dir / f"{e7_split_name}.run"
        run_e7_split(queries, reformulations_by_qid, descriptive_by_qid, bm25, reranker, doc_text_fn, out_path=e7_split_path, run_name=e7_split_name, rrf_k=rrf_k, depth=depth, config=config)
        out["E7-split"] = e7_split_path

    return out
