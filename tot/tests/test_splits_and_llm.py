"""Smoke tests for split guard and LLM cache / dry-run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tot.llm import LLMClient, LLMConfig, dry_run_estimate, make_cache_key
from tot.splits import assert_split_allowed, load_split


def test_test_split_blocked():
    with pytest.raises(ValueError, match="not allowed"):
        assert_split_allowed("test")


def test_dev2_blocked_by_default():
    with pytest.raises(ValueError, match="not allowed"):
        assert_split_allowed("dev2")


def test_dev2_allowed_with_a1():
    assert assert_split_allowed("dev2", allow_dev2=True) == "dev2"


def test_allowed_splits():
    for name in ("train", "dev1", "dev3", "DEV3"):
        assert assert_split_allowed(name) in {"train", "dev1", "dev3"}


def test_load_split_paths(tmp_path: Path):
    paths = load_split("dev3", data_dir=tmp_path)
    assert paths["queries"] == tmp_path / "dev3-queries.jsonl"
    assert paths["qrels"] == tmp_path / "dev3-qrels.txt"


def test_cache_key_stable():
    a = make_cache_key("stub", "m", "v0", "prompt {query}", 0.0, "hello")
    b = make_cache_key("stub", "m", "v0", "prompt {query}", 0.0, "hello")
    c = make_cache_key("stub", "m", "v0", "prompt {query}", 0.0, "hello!")
    assert a == b
    assert a != c


def test_llm_stub_caches_and_dry_run(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    client = LLMClient(config=LLMConfig(provider="stub", model="stub-model"), cache_dir=cache_dir)
    template = "Description: {query}"

    status = client.llm_call(
        task="decomposition",
        query="a film about dreams",
        prompt_version="stub_v0",
        qid="q1",
        prompt_template=template,
        dry_run=True,
    )
    assert status["would_call"] is True

    result = client.llm_call(
        task="decomposition",
        query="a film about dreams",
        prompt_version="stub_v0",
        qid="q1",
        prompt_template=template,
    )
    assert result.cached is False
    assert result.raw_output

    status2 = client.llm_call(
        task="decomposition",
        query="a film about dreams",
        prompt_version="stub_v0",
        qid="q1",
        prompt_template=template,
        dry_run=True,
    )
    assert status2["cached"] is True

    result2 = client.llm_call(
        task="decomposition",
        query="a film about dreams",
        prompt_version="stub_v0",
        qid="q1",
        prompt_template=template,
    )
    assert result2.cached is True

    summary = dry_run_estimate(
        client,
        task="decomposition",
        items=[{"qid": "q1", "query": "a film about dreams"}, {"qid": "q2", "query": "other"}],
        prompt_version="stub_v0",
        prompt_template=template,
    )
    assert summary["cached"] == 1
    assert summary["uncached"] == 1
    assert summary["estimated_api_calls"] == 1

    # Cache file written as JSONL
    rows = (cache_dir / "decompositions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["qid"] == "q1"
