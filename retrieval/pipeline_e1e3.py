"""
Run E1 / E2-R / E3 under Tier B (plan_v2/04).

E1: BM25(original)
E2-R: dense rerank of BM25 top-1000
E3: RRF(BM25 list, dense-reranked list)
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


def _sidecar(run_path: Path, payload: dict[str, Any]) -> None:
    side = run_path.with_suffix(".json")
    payload = {**payload, "created_utc": datetime.now(timezone.utc).isoformat()}
    side.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_e1(
    queries: list[dict[str, str]],
    bm25: Bm25Backend,
    *,
    out_path: str | Path,
    run_name: str,
    k: int = 1000,
    config: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for row in queries:
        hits = bm25.search(row["query"], k=k)
        rankings[row["qid"]] = [d for d, _ in hits]
        scores[row["qid"]] = [s for _, s in hits]
    write_trec_run(out_path, rankings, run_name=run_name, scores=scores)
    _sidecar(out_path, {"exp": "E1", "run_name": run_name, "config": config or {}, "n_queries": len(queries)})
    return out_path


def run_e2_r(
    queries: list[dict[str, str]],
    bm25_rankings: dict[str, list[str]],
    reranker: DenseReranker,
    doc_text_fn: DocTextFn,
    *,
    out_path: str | Path,
    run_name: str,
    top_k: int = 1000,
    config: dict[str, Any] | None = None,
) -> Path:
    """Dense reranking of BM25 candidates — label E2-R."""
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for row in queries:
        qid = row["qid"]
        cands = bm25_rankings.get(qid, [])[:top_k]
        hits = reranker.rerank(row["query"], cands, doc_text_fn, top_k=top_k)
        rankings[qid] = [d for d, _ in hits]
        scores[qid] = [s for _, s in hits]
    write_trec_run(out_path, rankings, run_name=run_name, scores=scores)
    _sidecar(
        out_path,
        {
            "exp": "E2-R",
            "run_name": run_name,
            "config": config or {},
            "n_queries": len(queries),
            "note": "dense reranking of BM25 candidates (Tier B); not full-corpus dense retrieval",
        },
    )
    return out_path


def run_e3(
    queries: list[dict[str, str]],
    bm25_rankings: dict[str, list[str]],
    dense_rankings: dict[str, list[str]],
    *,
    out_path: str | Path,
    run_name: str,
    rrf_k: int = 60,
    out_depth: int = 1000,
    config: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for row in queries:
        qid = row["qid"]
        fused = rrf_fuse_docids(
            [bm25_rankings.get(qid, []), dense_rankings.get(qid, [])],
            k=rrf_k,
            out_depth=out_depth,
        )
        rankings[qid] = fused
        # synthetic scores from fused rank
        scores[qid] = [float(len(fused) - i) for i in range(len(fused))]
    write_trec_run(out_path, rankings, run_name=run_name, scores=scores)
    _sidecar(out_path, {"exp": "E3", "run_name": run_name, "config": config or {}, "rrf_k": rrf_k})
    return out_path


def rankings_from_run_file(path: str | Path) -> dict[str, list[str]]:
    from evaluation.io_utils import load_run

    return load_run(path)


def run_e1_e2r_e3(
    query_path: str | Path,
    bm25: Bm25Backend,
    reranker: DenseReranker,
    doc_text_fn: DocTextFn,
    *,
    runs_dir: str | Path,
    split: str,
    corpus_tier: str = "dense_rerank_bm25",
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

    e1_name = make_run_name("E1", split, "tierB", "orig", date)
    e1_path = runs_dir / f"{e1_name}.run"
    run_e1(queries, bm25, out_path=e1_path, run_name=e1_name, k=depth, config=config)
    bm25_rankings = rankings_from_run_file(e1_path)

    e2_name = make_run_name("E2-R", split, "tierB", "orig", date)
    e2_path = runs_dir / f"{e2_name}.run"
    run_e2_r(
        queries,
        bm25_rankings,
        reranker,
        doc_text_fn,
        out_path=e2_path,
        run_name=e2_name,
        top_k=depth,
        config=config,
    )
    dense_rankings = rankings_from_run_file(e2_path)

    e3_name = make_run_name("E3", split, "tierB", "orig", date)
    e3_path = runs_dir / f"{e3_name}.run"
    run_e3(
        queries,
        bm25_rankings,
        dense_rankings,
        out_path=e3_path,
        run_name=e3_name,
        rrf_k=rrf_k,
        out_depth=depth,
        config=config,
    )
    return {"E1": e1_path, "E2-R": e2_path, "E3": e3_path, "corpus_tier": corpus_tier}  # type: ignore
