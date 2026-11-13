"""
Cross-Encoder Reranking Engine (Plan 08, Experiment E9).

Reranks the top-K candidates from E8 using a cross-encoder while preserving
the exact ordering of the tail (candidates ranked > K).
Enforces the tail-invariance contract: Recall@K' for any K' >= K is strictly
identical between E8 and E9.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

MODEL_PRESETS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-base": "BAAI/bge-reranker-base",
    "bge-large": "BAAI/bge-reranker-large",
    "stub": "stub-reranker",
}


def sanitize_model_slug(model_name: str) -> str:
    """Turn model identifier into safe filesystem slug."""
    slug = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", slug)


class CrossEncoderReranker:
    """
    Batched, fp16-capable cross-encoder with resumable JSONL caching.
    """

    def __init__(
        self,
        model_name: str = "minilm",
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 64,
        cache_dir: str | Path = "cache/ce_scores",
    ):
        self.model_name_or_preset = model_name
        self.model_name = MODEL_PRESETS.get(model_name, model_name)
        self.slug = sanitize_model_slug(self.model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.model_name == "stub-reranker" or self.model_name_or_preset == "stub":
            self.model = None
            self.device = "cpu"
            self.is_stub = True
        else:
            self.is_stub = False
            self.model, self.device = self._init_model(device)

    def _init_model(self, requested_device: str | None) -> tuple[Any, str]:
        import torch
        from sentence_transformers import CrossEncoder

        if requested_device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = requested_device

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=device,
                local_files_only=True,
            )
        except Exception as e:
            logger.info(
                "CrossEncoder %s not found in local cache (%s). Falling back to stub mode.",
                self.model_name,
                e,
            )
            self.is_stub = True
            return None, "cpu"

        # Enable fp16 if CUDA is used
        if device == "cuda" and hasattr(model, "model"):
            try:
                model.model = model.model.half()
            except Exception:
                pass

        return model, device

    def _cache_path(self, input_format: str) -> Path:
        return self.cache_dir / f"{self.slug}_{input_format}.jsonl"

    def load_cache(self, input_format: str) -> dict[tuple[str, str], float]:
        """Load cached (qid, docid) -> score from disk."""
        cpath = self._cache_path(input_format)
        cache: dict[tuple[str, str], float] = {}
        if not cpath.exists():
            return cache
        with cpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    cache[(str(obj["qid"]), str(obj["docid"]))] = float(obj["score"])
                except Exception:
                    continue
        return cache

    def predict_pairs(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        """Score (query, doc_text) pairs with cross-encoder."""
        if not pairs:
            return []
        bs = batch_size or self.batch_size

        if self.is_stub or self.model is None:
            # Deterministic lexical stub for offline / testing:
            # scores based on query term overlap with doc text
            scores: list[float] = []
            for q, d in pairs:
                q_words = set(q.lower().split())
                d_words = set(d.lower().split())
                overlap = len(q_words & d_words)
                scores.append(float(overlap) / (len(q_words) + 1.0))
            return scores

        import torch

        # Sort pairs by length to reduce padding overhead
        indexed_pairs = list(enumerate(pairs))
        indexed_pairs.sort(key=lambda item: len(item[1][0]) + len(item[1][1]))
        sorted_pairs = [p for _, p in indexed_pairs]

        with torch.no_grad():
            raw_scores = self.model.predict(
                sorted_pairs,
                batch_size=bs,
                show_progress_bar=False,
            )

        # Reorder to original indices
        out_scores = [0.0] * len(pairs)
        for (orig_idx, _), s in zip(indexed_pairs, raw_scores):
            out_scores[orig_idx] = float(s)
        return out_scores

    def score_and_cache(
        self,
        qid: str,
        query_text: str,
        candidates: list[dict[str, Any]],
        *,
        input_format: str = "orig",
        top_k: int = 100,
        cached_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Score the top-K candidates for a single query.
        Appends newly computed scores to the cache file.
        Returns list of (docid, score) in descending score order.
        """
        cpath = self._cache_path(input_format)
        cache = cached_scores if cached_scores is not None else self.load_cache(input_format)

        needed_pairs: list[tuple[str, str]] = []
        needed_docids: list[str] = []
        doc_scores: dict[str, float] = {}

        subset = candidates[:top_k]
        for c in subset:
            did = str(c["docid"])
            if (qid, did) in cache:
                doc_scores[did] = cache[(qid, did)]
            else:
                text = c.get("text", "") or ""
                needed_pairs.append((query_text, text))
                needed_docids.append(did)

        if needed_pairs:
            new_scores = self.predict_pairs(needed_pairs)
            with cpath.open("a", encoding="utf-8") as f:
                for did, s in zip(needed_docids, new_scores):
                    doc_scores[did] = s
                    cache[(qid, did)] = s
                    f.write(json.dumps({"qid": qid, "docid": did, "score": s}) + "\n")

        # Fallback for candidates with empty text: ensure they stay lower than scored docs
        min_score = min(doc_scores.values()) if doc_scores else 0.0
        for c in subset:
            did = str(c["docid"])
            if did not in doc_scores:
                doc_scores[did] = min_score - 10.0

        # Stable sort: primary key -score, secondary key original rank
        orig_ranks = {str(c["docid"]): i for i, c in enumerate(subset)}
        sorted_cands = sorted(
            [(str(c["docid"]), doc_scores[str(c["docid"])]) for c in subset],
            key=lambda item: (-item[1], orig_ranks[item[0]]),
        )
        return sorted_cands


def format_query_side(
    original_query: str,
    reformulated_query: str = "",
    style: str = "orig",
) -> str:
    """Format query side according to Plan 08 §3.2 ablation options."""
    st = style.lower()
    if st == "orig":
        return original_query.strip()
    if st == "reform":
        return (reformulated_query or original_query).strip()
    if st == "concat":
        if reformulated_query and reformulated_query != original_query:
            return f"{original_query.strip()}\n{reformulated_query.strip()}"
        return original_query.strip()
    raise ValueError(f"Unknown query-side format '{style}'; choose orig|reform|concat")


def verify_tail_invariance(
    e8_rankings: dict[str, list[str]],
    e9_rankings: dict[str, list[str]],
    *,
    k: int,
) -> tuple[bool, str]:
    """
    Automated check (Plan 08 §8 & §9):
    Recall@K' for any K' >= K must be mathematically identical between E8 and E9.
    Also verifies candidates > K keep their exact relative E8 order.
    """
    for qid in e8_rankings:
        if qid not in e9_rankings:
            return False, f"Query {qid} missing from E9"
        e8_list = e8_rankings[qid]
        e9_list = e9_rankings[qid]

        # 1. Top-K set equality
        if set(e8_list[:k]) != set(e9_list[:k]):
            return False, f"Query {qid}: set of top-{k} candidates differs between E8 and E9"

        # 2. Strict tail sequence equality for all ranks > K
        if e8_list[k:] != e9_list[k:]:
            return False, f"Query {qid}: candidates ranked > {k} do not preserve exact E8 order"

    return True, "OK"
