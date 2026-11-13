"""Reciprocal Rank Fusion (plan_v2/04 §4.1)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
    out_depth: int = 1000,
) -> list[tuple[str, float]]:
    """
    Fuse N ranked lists with unweighted RRF.

    RRF(d) = sum over lists L containing d of  1 / (k + rank_L(d))
    Ranks are 1-based. Ties: higher RRF first, then smaller best rank, then docid.
    Empty lists are ignored. Documents missing from a list contribute nothing from it.
    """
    if k < 0:
        raise ValueError("k must be >= 0")

    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}

    for lst in ranked_lists:
        if not lst:
            continue
        seen_in_list: set[str] = set()
        for rank, docid in enumerate(lst, start=1):
            docid = str(docid)
            if docid in seen_in_list:
                continue
            seen_in_list.add(docid)
            scores[docid] += 1.0 / (k + rank)
            prev = best_rank.get(docid)
            if prev is None or rank < prev:
                best_rank[docid] = rank

    ranked = sorted(
        scores.keys(),
        key=lambda d: (-scores[d], best_rank.get(d, 10**9), d),
    )
    out = ranked[:out_depth]
    return [(d, scores[d]) for d in out]


def rrf_fuse_docids(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
    out_depth: int = 1000,
) -> list[str]:
    return [d for d, _ in rrf_fuse(ranked_lists, k=k, out_depth=out_depth)]
