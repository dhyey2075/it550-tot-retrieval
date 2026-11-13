#!/usr/bin/env python3
"""
Decomposition plumbing on Dev1 (plan 05).

Uses LLM stub unless OPENAI_API_KEY / GEMINI_API_KEY is set.
Always writes cache + facet non-N/A rates. Does not invent titles.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.queries import load_queries
from tot.decomposition import FACET_KEYS, non_na_rates, parse_facets
from tot.llm import LLMClient, LLMConfig


PROMPT_VERSION = "decomposition_extractive_v1"


def pick_provider() -> LLMConfig:
    if os.environ.get("OPENAI_API_KEY"):
        return LLMConfig(provider="openai", model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return LLMConfig(provider="gemini", model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    return LLMConfig(provider="stub", model="stub-model")


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "dev1"
    queries = load_queries(ROOT / "data" / f"{split}-queries.jsonl")
    prompt_tmpl = (ROOT / "prompts" / "decomposition_extractive.txt").read_text(encoding="utf-8")

    cfg = pick_provider()
    client = LLMClient(config=cfg, cache_dir=ROOT / "cache")
    print(f"[progress] provider={cfg.provider} model={cfg.model} n={len(queries)}", flush=True)

    facets_out = ROOT / "cache" / f"facets_{split}_extractive.jsonl"
    if facets_out.exists():
        facets_out.unlink()

    facet_rows: list[dict[str, str]] = []
    t0 = time.time()
    n_cached = 0
    for i, row in enumerate(queries, 1):
        result = client.llm_call(
            task="decomposition",
            query=row["query"],
            prompt_version=PROMPT_VERSION,
            qid=row["qid"],
            prompt_template=prompt_tmpl,
        )
        if result.cached:
            n_cached += 1
        raw = result.raw_output
        try:
            if isinstance(result.parsed, dict) and all(k in result.parsed for k in FACET_KEYS[:1]):
                # prefer structured parse when available
                facets = parse_facets(json.dumps(result.parsed) if not isinstance(result.parsed, str) else result.parsed)
            else:
                facets = parse_facets(raw, predictive=False)
        except Exception:
            try:
                facets = parse_facets(raw, predictive=False)
            except Exception as exc:
                facets = {k: "N/A" for k in FACET_KEYS}
                facets["_parse_error"] = str(exc)
        # keep only facet keys for rates
        clean = {k: facets.get(k, "N/A") for k in FACET_KEYS}
        facet_rows.append(clean)
        with facets_out.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "qid": row["qid"],
                        "facets": clean,
                        "cached": result.cached,
                        "provider": cfg.provider,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if i % 20 == 0 or i == 1:
            print(
                f"  decomp {i}/{len(queries)} cache_hits={n_cached} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    rates = non_na_rates(facet_rows)
    summary = {
        "split": split,
        "provider": cfg.provider,
        "model": cfg.model,
        "n": len(queries),
        "cache_hits": n_cached,
        "non_na_rates": rates,
        "note": (
            "stub facets are plumbing-only; do not use for Dev3 claims"
            if cfg.provider == "stub"
            else "live LLM facets"
        ),
    }
    sum_path = ROOT / "logs" / f"decomp_{split}_summary.json"
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[progress] wrote {facets_out} and {sum_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
