"""Unit tests for Plan 08 Cross-Encoder Reranking and Invariant Enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reranking.cross_encoder import (
    CrossEncoderReranker,
    format_query_side,
    sanitize_model_slug,
    verify_tail_invariance,
)


def test_sanitize_model_slug():
    assert sanitize_model_slug("cross-encoder/ms-marco-MiniLM-L-6-v2") == "ms-marco-MiniLM-L-6-v2"
    assert sanitize_model_slug("BAAI/bge-reranker-base") == "bge-reranker-base"


def test_format_query_side():
    orig = "movie about dreaming inside a dream"
    reform = "Inception sci-fi film Christopher Nolan"

    assert format_query_side(orig, reform, "orig") == orig
    assert format_query_side(orig, reform, "reform") == reform
    assert format_query_side(orig, reform, "concat") == f"{orig}\n{reform}"
    assert format_query_side(orig, "", "reform") == orig


def test_tail_invariance_passes():
    e8 = {
        "q1": [f"D{i}" for i in range(1, 101)],
        "q2": [f"D{i}" for i in range(101, 201)],
    }
    # E9 shuffles top 20, keeps 21..100 unchanged
    e9 = {
        "q1": list(reversed(e8["q1"][:20])) + e8["q1"][20:],
        "q2": list(reversed(e8["q2"][:20])) + e8["q2"][20:],
    }
    ok, msg = verify_tail_invariance(e8, e9, k=20)
    assert ok, msg


def test_tail_invariance_fails_on_leak():
    e8 = {"q1": [f"D{i}" for i in range(1, 101)]}
    # A candidate from the tail (D30) leaked into the top-20
    e9 = {"q1": ["D30"] + e8["q1"][:19] + e8["q1"][20:]}
    ok, _ = verify_tail_invariance(e8, e9, k=20)
    assert not ok, "Invariant check must catch candidates leaking between top-K and tail"


def test_cross_encoder_caching_and_scoring(tmp_path: Path):
    reranker = CrossEncoderReranker(model_name="stub", cache_dir=tmp_path)
    cands = [
        {"docid": "D1", "text": "apple banana cherry"},
        {"docid": "D2", "text": "banana cherry date"},
        {"docid": "D3", "text": ""},
    ]
    query = "apple fruit"

    # Score candidates
    scored = reranker.score_and_cache("q1", query, cands, input_format="orig", top_k=3)
    assert len(scored) == 3
    # D1 has word 'apple' matching query, so stub should rank D1 first
    assert scored[0][0] == "D1"

    # Verify cache file was written
    cpath = tmp_path / "stub-reranker_orig.jsonl"
    assert cpath.exists()

    # Load cache directly and check values
    cache = reranker.load_cache("orig")
    assert ("q1", "D1") in cache
    assert ("q1", "D2") in cache
