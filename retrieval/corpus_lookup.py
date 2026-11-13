"""Random-access / cached lookup of document text for Tier-B dense reranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from retrieval.docrep import doc_id, doc_text


class JsonlCorpusCache:
    """
    Load a JSONL sample (or sub-corpus) into memory keyed by id.

    For the full 6.4M corpus on Colab, prefer offsets + seek, or fetch text
    from Pyserini `searcher.doc(docid).raw()` when using Anserini.
    """

    def __init__(self, path: str | Path, *, lead_chars: int = 1000):
        self.lead_chars = lead_chars
        self.by_id: dict[str, dict[str, Any]] = {}
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.by_id[doc_id(obj)] = obj

    def text(self, docid: str) -> str:
        doc = self.by_id.get(str(docid))
        if doc is None:
            return ""
        return doc_text(doc, lead_chars=self.lead_chars)

    def __contains__(self, docid: str) -> bool:
        return str(docid) in self.by_id

    def __len__(self) -> int:
        return len(self.by_id)


class PyseriniDocStore:
    """Fetch stored raw JSON from a Lucene/Anserini index when available."""

    def __init__(self, searcher, *, lead_chars: int = 1000):
        self.searcher = searcher
        self.lead_chars = lead_chars
        self._cache: dict[str, str] = {}

    def text(self, docid: str) -> str:
        docid = str(docid)
        if docid in self._cache:
            return self._cache[docid]
        try:
            raw = self.searcher.doc(docid).raw()
            obj = json.loads(raw) if raw and raw.lstrip().startswith("{") else {"title": "", "text": raw or ""}
            # Some indexes store contents differently
            if "contents" in obj and "text" not in obj:
                obj["text"] = obj["contents"]
            t = doc_text(obj, lead_chars=self.lead_chars)
        except Exception:
            t = ""
        self._cache[docid] = t
        return t
