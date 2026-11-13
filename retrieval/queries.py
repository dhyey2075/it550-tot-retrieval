"""Load TREC-ToT query JSONL (query_id, query)."""

from __future__ import annotations

import json
from pathlib import Path


def load_queries(path: str | Path) -> list[dict[str, str]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id") or obj.get("qid") or "")
            query = obj.get("query") or ""
            if not isinstance(query, str):
                query = str(query)
            rows.append({"qid": qid, "query": query})
    return rows
