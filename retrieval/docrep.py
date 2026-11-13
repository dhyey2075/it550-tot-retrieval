"""Document text representation shared by BM25 and dense (feeds D-02/D-03)."""

from __future__ import annotations

from typing import Any, Mapping


def doc_text(
    doc: Mapping[str, Any],
    *,
    lead_chars: int = 1000,
    include_url: bool = False,
) -> str:
    """
    Build the indexed/encoded string for a corpus record.

    On-disk fields are id/url/title/text (confirmed in data_report.md).
    Working default: title + first `lead_chars` of text.
    """
    title = str(doc.get("title") or "").strip()
    text = str(doc.get("text") or "")
    lead = text[:lead_chars]
    parts = [p for p in (title, lead) if p]
    if include_url:
        url = str(doc.get("url") or "").strip()
        if url:
            parts.append(url)
    return "\n".join(parts).strip()


def doc_id(doc: Mapping[str, Any]) -> str:
    if "id" in doc:
        return str(doc["id"])
    if "doc_id" in doc:
        return str(doc["doc_id"])
    raise KeyError("document missing id/doc_id")
