#!/usr/bin/env python3
"""Run Dev1 query reformulation experiments: E7 and diagnostics (Plan 06)."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_run import main as eval_main
from retrieval.bm25.search import PyseriniSearcher
from retrieval.corpus_lookup import JsonlCorpusCache, PyseriniDocStore
from retrieval.dense.rerank import build_dense_reranker
from retrieval.pipeline_e7 import run_e7_suite
from retrieval.queries import load_queries
from tot.llm import LLMClient, LLMConfig
from tot.reformulation import audit_reformulations, clean_reformulation, load_cached_reformulations


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


def pick_provider() -> LLMConfig:
    if os.environ.get("OPENAI_API_KEY"):
        return LLMConfig(provider="openai", model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return LLMConfig(provider="gemini", model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    return LLMConfig(provider="stub", model="stub-model")


def ensure_reformulations(
    queries: list[dict[str, str]],
    split: str,
    style: str = "keyword",
) -> dict[str, str]:
    """Ensure reformulations exist in cache; if missing, generate via LLM (or stub)."""
    cache_path = ROOT / "cache" / f"reformulations_{split}_{style}.jsonl"
    reforms = load_cached_reformulations(cache_path)
    if len(reforms) >= len(queries):
        return reforms

    cfg = pick_provider()
    client = LLMClient(config=cfg, cache_dir=ROOT / "cache")
    prompt_file = ROOT / "prompts" / f"reformulation_{style}.txt"
    if not prompt_file.exists():
        prompt_file = ROOT / "prompts" / "reformulation_keyword.txt"
    prompt_tmpl = prompt_file.read_text(encoding="utf-8")
    version = f"reform_{style}_v0"

    print(f"[progress] generating {style} reformulations ({cfg.provider}/{cfg.model})...", flush=True)
    out: dict[str, str] = {}
    with cache_path.open("a", encoding="utf-8") as f:
        for row in queries:
            qid = row["qid"]
            if qid in reforms:
                out[qid] = reforms[qid]
                continue
            res = client.llm_call(
                task="reformulation",
                query=row["query"],
                prompt_version=version,
                qid=qid,
                prompt_template=prompt_tmpl,
            )
            clean = clean_reformulation(res.raw_output, fallback=row["query"])
            out[qid] = clean
            f.write(
                json.dumps(
                    {
                        "qid": qid,
                        "style": style,
                        "reformulation": clean,
                        "provider": cfg.provider,
                        "query_text": row["query"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out


def main() -> int:
    split = "dev1"
    queries = load_queries(ROOT / "data" / f"{split}-queries.jsonl")
    qrels_path = ROOT / "data" / f"{split}-qrels.txt"
    index_dir = find_index(ROOT / "indices" / "anserini")

    print(f"[progress] loaded {len(queries)} {split} queries", flush=True)

    # 1. Obtain reformulations (both keyword R-K and descriptive R-N for diagnostics)
    kw_reforms = ensure_reformulations(queries, split, style="keyword")
    desc_reforms = ensure_reformulations(queries, split, style="descriptive")

    # Run and log audit
    orig_map = {r["qid"]: r["query"] for r in queries}
    kw_audit = audit_reformulations(kw_reforms, orig_map)
    print(f"[audit kw] mean_words={kw_audit.get('mean_words',0):.1f} fallback_rate={kw_audit.get('fallback_rate',0):.2f}")

    bm25 = PyseriniSearcher(index_dir)
    cand_jsonl = ROOT / "data" / "corpus" / "dev1_e1_candidates.jsonl"
    if cand_jsonl.exists() and cand_jsonl.stat().st_size > 0:
        doc_store = JsonlCorpusCache(cand_jsonl, lead_chars=1000)
    else:
        doc_store = PyseriniDocStore(bm25.searcher, lead_chars=1000)

    print("[progress] initializing e5-small reranker on MPS/CUDA...", flush=True)
    reranker = build_dense_reranker("e5-small")

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "corpus_tier": "dense_rerank_bm25",
        "bm25": {"backend": "pyserini", "k1": None, "b": None, "lead_chars": 1000},
        "dense": {"preset": "e5-small", "role": "rerank_bm25_candidates"},
        "rrf_k": 60,
        "reformulation_style": "keyword",
    }

    t0 = time.time()
    print("[progress] running E7 suite (E7, E7-bm25, E7-dense, E7-split)...", flush=True)
    paths = run_e7_suite(
        ROOT / "data" / f"{split}-queries.jsonl",
        kw_reforms,
        bm25,
        reranker,
        doc_store.text,
        runs_dir=runs_dir,
        split=split,
        style="keyword",
        descriptive_by_qid=None,
        rrf_k=60,
        depth=1000,
        date=date,
        config=cfg,
    )
    print(f"[progress] E7 suite runs completed in {time.time()-t0:.1f}s", flush=True)

    # Evaluate each run
    for exp, path in paths.items():
        print(f"[progress] evaluating {exp} ({path.name})", flush=True)
        eval_main(
            [
                str(path),
                "--exp",
                exp,
                "--qrels",
                str(qrels_path),
                "--split",
                split,
                "--corpus-tier",
                "dense_rerank_bm25",
                "--results",
                str(ROOT / "evaluation" / "results.csv"),
                "--per-query-dir",
                str(ROOT / "evaluation" / "per_query"),
            ]
        )

    print("[progress] Dev1 E7 suite evaluated successfully!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
