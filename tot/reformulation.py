"""Query reformulation helpers, parsing, cleaning and caching (plan_v2/06)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_PREFIX_PATTERNS = [
    r"^\[stub [^\]]+\]\s*",
    r"^here is (?:the|a) (?:search query|query|rewrite|summary):\s*",
    r"^search query:\s*",
    r"^query:\s*",
    r"^reformulation:\s*",
    r"^rewrite:\s*",
]


def clean_reformulation(raw: str, *, max_words: int = 80, fallback: str = "") -> str:
    """Clean and standardize raw LLM reformulation text."""
    if not raw or not raw.strip():
        return fallback.strip()

    t = raw.strip()
    # Strip markdown fences
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
        t = t.strip()

    # Strip conversational prefixes
    for pat in _PREFIX_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()

    # Strip surrounding quotes if wrapped
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()

    # Collapse multiple whitespaces/newlines into single spaces
    t = re.sub(r"\s+", " ", t).strip()

    # Cap word count
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]).strip()

    if not t:
        return fallback.strip()
    return t


def load_cached_reformulations(cache_path: str | Path) -> dict[str, str]:
    """Load cached reformulations from JSONL file into a {qid: rewritten_query} dict."""
    out: dict[str, str] = {}
    path = Path(cache_path)
    if not path.exists():
        return out

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or "")
            if not qid:
                continue

            raw = row.get("raw_output") or row.get("reformulation") or row.get("parsed") or ""
            if isinstance(raw, dict):
                raw = raw.get("query") or raw.get("reformulation") or str(raw)
            clean = clean_reformulation(str(raw), fallback=str(row.get("query_text") or ""))
            if clean:
                out[qid] = clean
    return out


def audit_reformulations(
    reformulations: dict[str, str],
    original_queries: dict[str, str],
    target_titles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Automated diagnostics & audit metrics:
    - average word count
    - fallback rate (reformulation == original)
    - introduced entities heuristic (capitalized non-stopwords absent from original)
    - target title token overlap audit (analysis-only, post-hoc)
    """
    if not reformulations:
        return {}

    n = len(reformulations)
    word_counts = []
    fallbacks = 0
    novel_entities_count = 0
    title_matches = 0

    for qid, reform in reformulations.items():
        orig = original_queries.get(qid, "")
        words = reform.split()
        word_counts.append(len(words))
        if reform.strip() == orig.strip():
            fallbacks += 1

        # Check novel capitalized words not in original
        orig_words_lower = set(re.findall(r"\b\w+\b", orig.lower()))
        reform_tokens = re.findall(r"\b[A-Z][a-z0-9]+\b", reform)
        novel = [w for w in reform_tokens if w.lower() not in orig_words_lower]
        if novel:
            novel_entities_count += 1

        # Title match audit (if target titles provided)
        if target_titles and qid in target_titles:
            title = target_titles[qid].lower().strip()
            if title and title in reform.lower():
                title_matches += 1

    return {
        "n_queries": n,
        "mean_words": sum(word_counts) / n if n else 0.0,
        "max_words": max(word_counts) if word_counts else 0,
        "min_words": min(word_counts) if word_counts else 0,
        "fallback_rate": fallbacks / n if n else 0.0,
        "novel_entity_query_rate": novel_entities_count / n if n else 0.0,
        "title_exact_match_rate": (title_matches / n) if target_titles and n else 0.0,
    }
