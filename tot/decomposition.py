"""Facet-based query decomposition (plan_v2/05)."""

from __future__ import annotations

import json
import re
from typing import Any

FACET_KEYS = [
    "category",
    "description",
    "people",
    "place_setting",
    "time_period",
    "distinctive_details",
    "name_fragments",
]

PREDICTIVE_KEYS = [k for k in FACET_KEYS if k != "name_fragments"]

_NA = {"n/a", "na", "none", "null", ""}


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_facets(raw: str, *, predictive: bool = False) -> dict[str, str]:
    keys = PREDICTIVE_KEYS if predictive else FACET_KEYS
    cleaned = _strip_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # try to find a JSON object substring
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("facet payload is not a JSON object")
    out: dict[str, str] = {}
    for k in keys:
        v = obj.get(k, "N/A")
        if v is None:
            v = "N/A"
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if v.lower() in _NA:
            v = "N/A"
        # length cap per facet
        if len(v) > 300:
            v = v[:300].rstrip()
        out[k] = v
    return out


def active_facets(facets: dict[str, str]) -> list[tuple[str, str]]:
    """Non-N/A facets as (name, text), deduplicated by text."""
    seen: set[str] = set()
    active: list[tuple[str, str]] = []
    for k, v in facets.items():
        if v == "N/A":
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        active.append((k, v))
    return active


def facet_queries(facets: dict[str, str], *, fallback_query: str) -> list[str]:
    active = active_facets(facets)
    if not active:
        return [fallback_query]
    return [text for _name, text in active]


def concat_facets(facets: dict[str, str], *, fallback_query: str) -> str:
    parts = [text for _n, text in active_facets(facets)]
    return " ".join(parts) if parts else fallback_query


def non_na_rates(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    rates = {}
    n = len(rows)
    for k in sorted(keys):
        rates[k] = sum(1 for r in rows if r.get(k, "N/A") != "N/A") / n
    return rates
