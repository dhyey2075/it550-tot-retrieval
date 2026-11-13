"""
Decomposition retrieval: E4 / E5-R / E6 under Tier B (plan_v2/05).

E4: RRF over BM25(facet_i)
E5-R: for each facet, dense-rerank that facet's BM25 top-1000; RRF over those lists
E6: flat RRF over all BM25 facet lists + dense-reranked facet lists (default protocol)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation.io_utils import write_trec_run
from retrieval.bm25.search import Bm25Backend
from retrieval.dense.rerank import DenseReranker
from retrieval.rrf import rrf_fuse_docids
from tot.decomposition import active_facets, concat_facets, facet_queries, parse_facets

DocTextFn = Callable[[str], str]


def _sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps({**payload, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_cached_facets(cache_path: str | Path) -> dict[str, dict[str, str]]:
    """Load parsed facets from LLM cache JSONL (expects parsed dict or raw_output)."""
    out: dict[str, dict[str, str]] = {}
    path = Path(cache_path)
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or "")
            parsed = row.get("parsed") or row.get("facets")
            if isinstance(parsed, dict) and parsed:
                try:
                    # may already be facets or {"clues":...}; prefer facet keys
                    if "category" in parsed or "description" in parsed:
                        out[qid] = {k: str(parsed.get(k, "N/A")) for k in parsed}
                        continue
                except Exception:
                    pass
            raw = row.get("raw_output") or ""
            if raw:
                try:
                    out[qid] = parse_facets(raw, predictive=False)
                except Exception:
                    continue
    return out


def run_e4(
    queries: list[dict[str, str]],
    facets_by_qid: dict[str, dict[str, str]],
    bm25: Bm25Backend,
    *,
    out_path: str | Path,
    run_name: str,
    rrf_k: int = 60,
    depth: int = 1000,
    include_original: bool = False,
    config: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    fallbacks = 0
    for row in queries:
        qid, query = row["qid"], row["query"]
        facets = facets_by_qid.get(qid) or {}
        fqueries = facet_queries(facets, fallback_query=query)
        if fqueries == [query]:
            fallbacks += 1
        lists = [ [d for d, _ in bm25.search(fq, k=depth)] for fq in fqueries ]
        if include_original:
            lists.append([d for d, _ in bm25.search(query, k=depth)])
        rankings[qid] = rrf_fuse_docids(lists, k=rrf_k, out_depth=depth)
    write_trec_run(out_path, rankings, run_name=run_name)
    _sidecar(out_path, {"exp": "E4" + ("+" if include_original else ""), "fallbacks": fallbacks, "config": config or {}})
    return out_path


def run_e5_r(
    queries: list[dict[str, str]],
    facets_by_qid: dict[str, dict[str, str]],
    bm25: Bm25Backend,
    reranker: DenseReranker,
    doc_text_fn: DocTextFn,
    *,
    out_path: str | Path,
    run_name: str,
    rrf_k: int = 60,
    depth: int = 1000,
    include_original: bool = False,
    config: dict[str, Any] | None = None,
) -> Path:
    """Per-facet BM25 → dense rerank → RRF. Label E5-R."""
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    for row in queries:
        qid, query = row["qid"], row["query"]
        facets = facets_by_qid.get(qid) or {}
        fqueries = facet_queries(facets, fallback_query=query)
        lists: list[list[str]] = []
        for fq in fqueries:
            bm25_hits = [d for d, _ in bm25.search(fq, k=depth)]
            dense_hits = reranker.rerank(fq, bm25_hits, doc_text_fn, top_k=depth)
            lists.append([d for d, _ in dense_hits])
        if include_original:
            bm25_hits = [d for d, _ in bm25.search(query, k=depth)]
            dense_hits = reranker.rerank(query, bm25_hits, doc_text_fn, top_k=depth)
            lists.append([d for d, _ in dense_hits])
        rankings[qid] = rrf_fuse_docids(lists, k=rrf_k, out_depth=depth)
    write_trec_run(out_path, rankings, run_name=run_name)
    _sidecar(
        out_path,
        {
            "exp": "E5-R" + ("+" if include_original else ""),
            "note": "dense rerank of each facet's BM25 candidates (Tier B)",
            "config": config or {},
        },
    )
    return out_path


def run_e6(
    queries: list[dict[str, str]],
    facets_by_qid: dict[str, dict[str, str]],
    bm25: Bm25Backend,
    reranker: DenseReranker,
    doc_text_fn: DocTextFn,
    *,
    out_path: str | Path,
    run_name: str,
    rrf_k: int = 60,
    depth: int = 1000,
    protocol: str = "flat",
    include_original: bool = False,
    config: dict[str, Any] | None = None,
) -> Path:
    """
    protocol:
      - flat: RRF over all BM25 facet lists + all dense-reranked facet lists
      - two_level: per-facet RRF(BM25, dense) then RRF across facets
    """
    out_path = Path(out_path)
    rankings: dict[str, list[str]] = {}
    for row in queries:
        qid, query = row["qid"], row["query"]
        facets = facets_by_qid.get(qid) or {}
        fqueries = facet_queries(facets, fallback_query=query)
        bm25_lists: list[list[str]] = []
        dense_lists: list[list[str]] = []
        per_facet_fused: list[list[str]] = []
        for fq in fqueries:
            b_list = [d for d, _ in bm25.search(fq, k=depth)]
            d_list = [d for d, _ in reranker.rerank(fq, b_list, doc_text_fn, top_k=depth)]
            bm25_lists.append(b_list)
            dense_lists.append(d_list)
            per_facet_fused.append(rrf_fuse_docids([b_list, d_list], k=rrf_k, out_depth=depth))
        if include_original:
            b_list = [d for d, _ in bm25.search(query, k=depth)]
            d_list = [d for d, _ in reranker.rerank(query, b_list, doc_text_fn, top_k=depth)]
            bm25_lists.append(b_list)
            dense_lists.append(d_list)
            per_facet_fused.append(rrf_fuse_docids([b_list, d_list], k=rrf_k, out_depth=depth))

        if protocol == "two_level":
            rankings[qid] = rrf_fuse_docids(per_facet_fused, k=rrf_k, out_depth=depth)
        else:
            rankings[qid] = rrf_fuse_docids(bm25_lists + dense_lists, k=rrf_k, out_depth=depth)
    write_trec_run(out_path, rankings, run_name=run_name)
    _sidecar(out_path, {"exp": "E6", "protocol": protocol, "include_original": include_original, "config": config or {}})
    return out_path


def run_e4c(
    queries: list[dict[str, str]],
    facets_by_qid: dict[str, dict[str, str]],
    bm25: Bm25Backend,
    *,
    out_path: str | Path,
    run_name: str,
    depth: int = 1000,
) -> Path:
    """Concatenation baseline: one BM25 query from joined facets."""
    out_path = Path(out_path)
    rankings = {}
    for row in queries:
        qid, query = row["qid"], row["query"]
        cq = concat_facets(facets_by_qid.get(qid) or {}, fallback_query=query)
        rankings[qid] = [d for d, _ in bm25.search(cq, k=depth)]
    write_trec_run(out_path, rankings, run_name=run_name)
    _sidecar(out_path, {"exp": "E4c"})
    return out_path


def run_e4_e5r_e6(
    query_path: str | Path,
    facets_by_qid: dict[str, dict[str, str]],
    bm25: Bm25Backend,
    reranker: DenseReranker,
    doc_text_fn: DocTextFn,
    *,
    runs_dir: str | Path,
    split: str,
    rrf_k: int = 60,
    depth: int = 1000,
    date: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    from tot.run_naming import make_run_name
    from retrieval.queries import load_queries

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
    queries = load_queries(query_path)
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    e4_name = make_run_name("E4", split, "tierB", "extractive", date)
    e4_path = runs_dir / f"{e4_name}.run"
    run_e4(queries, facets_by_qid, bm25, out_path=e4_path, run_name=e4_name, rrf_k=rrf_k, depth=depth, config=config)

    e4c_name = make_run_name("E4c", split, "tierB", "concat", date)
    e4c_path = runs_dir / f"{e4c_name}.run"
    run_e4c(queries, facets_by_qid, bm25, out_path=e4c_path, run_name=e4c_name, depth=depth)

    e5_name = make_run_name("E5-R", split, "tierB", "extractive", date)
    e5_path = runs_dir / f"{e5_name}.run"
    run_e5_r(queries, facets_by_qid, bm25, reranker, doc_text_fn, out_path=e5_path, run_name=e5_name, rrf_k=rrf_k, depth=depth, config=config)

    e6_name = make_run_name("E6", split, "tierB", "extractive", date)
    e6_path = runs_dir / f"{e6_name}.run"
    run_e6(queries, facets_by_qid, bm25, reranker, doc_text_fn, out_path=e6_path, run_name=e6_name, rrf_k=rrf_k, depth=depth, config=config)

    return {"E4": e4_path, "E4c": e4c_path, "E5-R": e5_path, "E6": e6_path}
