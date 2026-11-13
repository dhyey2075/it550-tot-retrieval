"""Unit tests for RRF and facet decomposition parsing."""

from __future__ import annotations

import pytest

from retrieval.rrf import rrf_fuse, rrf_fuse_docids
from tot.decomposition import active_facets, concat_facets, facet_queries, parse_facets


def test_rrf_two_lists_known():
    # doc A rank1 in L1 and rank2 in L2; doc B rank2 in L1 and rank1 in L2
    L1 = ["A", "B", "C"]
    L2 = ["B", "A", "D"]
    fused = rrf_fuse([L1, L2], k=60, out_depth=10)
    docids = [d for d, _ in fused]
    # B and A should be top; with k=60, B: 1/61+1/62, A: 1/61+1/62 — equal RRF
    # tie-break: smaller best rank — both have best rank 1 → then docid
    assert set(docids[:2]) == {"A", "B"}
    assert docids[0] < docids[1] or fused[0][1] >= fused[1][1]


def test_rrf_doc_in_one_list_only():
    fused = rrf_fuse_docids([["A", "B"], ["C"]], k=60)
    assert "A" in fused and "C" in fused


def test_rrf_identical_lists():
    L = ["A", "B", "C"]
    assert rrf_fuse_docids([L, L], k=60)[:3] == ["A", "B", "C"]


def test_rrf_empty_list():
    assert rrf_fuse_docids([[], ["A", "B"]], k=60)[:2] == ["A", "B"]
    assert rrf_fuse_docids([[], []], k=60) == []


def test_rrf_accepts_n_lists():
    lists = [[f"d{i}_{j}" for j in range(5)] for i in range(7)]
    lists[0][0] = "TARGET"
    for i in range(1, 7):
        lists[i][2] = "TARGET"
    out = rrf_fuse_docids(lists, k=60, out_depth=10)
    assert out[0] == "TARGET"


def test_parse_facets_json():
    raw = '{"category":"film","description":"dreams","people":"N/A","place_setting":"N/A","time_period":"N/A","distinctive_details":"idea","name_fragments":"N/A"}'
    facets = parse_facets(raw)
    assert facets["category"] == "film"
    assert active_facets(facets) == [("category", "film"), ("description", "dreams"), ("distinctive_details", "idea")]
    assert facet_queries(facets, fallback_query="ORIG") == ["film", "dreams", "idea"]


def test_parse_facets_fences_and_fallback():
    raw = "```json\n{\"category\":\"N/A\",\"description\":\"N/A\",\"people\":\"N/A\",\"place_setting\":\"N/A\",\"time_period\":\"N/A\",\"distinctive_details\":\"N/A\",\"name_fragments\":\"N/A\"}\n```"
    facets = parse_facets(raw)
    assert facet_queries(facets, fallback_query="ORIG") == ["ORIG"]
    assert concat_facets(facets, fallback_query="ORIG") == "ORIG"


def test_bm25s_smoke_if_sample_exists():
    from pathlib import Path

    sample = Path("data/samples/corpus_sample_10k.jsonl")
    if not sample.exists():
        pytest.skip("sample corpus missing")
    from retrieval.bm25.search import Bm25sSearcher

    # tiny subset via reading first 200 lines
    import json

    docs = []
    with sample.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            docs.append(json.loads(line))
    searcher = Bm25sSearcher(docs=docs, lead_chars=500)
    hits = searcher.search("anarchism political philosophy", k=5)
    assert hits
    assert all(isinstance(d, str) and isinstance(s, float) for d, s in hits)


def test_clean_reformulation():
    from tot.reformulation import clean_reformulation

    raw1 = 'Here is the search query: "science-fiction film shared dreams Christopher Nolan"'
    assert clean_reformulation(raw1) == "science-fiction film shared dreams Christopher Nolan"

    raw2 = "```\nquery: suspension bridge San Francisco red towers\n```"
    assert clean_reformulation(raw2) == "suspension bridge San Francisco red towers"

    raw3 = ""
    assert clean_reformulation(raw3, fallback="fallback query") == "fallback query"


def test_weighted_rrf_fuse():
    from retrieval.hybrid.e8_fusion import weighted_rrf_fuse

    list1 = ["D1", "D2", "D3"]
    list2 = ["D2", "D1", "D3"]
    # If list2 is weighted 5x more, D2 should rank first
    fused = weighted_rrf_fuse([(list1, 1.0), (list2, 5.0)], k=60)
    assert fused[0] == "D2"


def test_audit_reformulations():
    from tot.reformulation import audit_reformulations

    reforms = {"1": "science fiction movie with dreams", "2": "suspension bridge in California"}
    origs = {"1": "I remember a movie about dreams", "2": "a bridge in California"}
    titles = {"1": "Inception", "2": "Golden Gate Bridge"}
    audit = audit_reformulations(reforms, origs, titles)
    assert audit["n_queries"] == 2
    assert audit["fallback_rate"] == 0.0
    assert audit["title_exact_match_rate"] == 0.0

