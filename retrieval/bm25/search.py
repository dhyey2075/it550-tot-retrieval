"""BM25 search backends: Pyserini (full corpus) and bm25s (samples / Tier C)."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence

from retrieval.docrep import doc_id, doc_text

logger = logging.getLogger(__name__)

# Conservative cap for extremely long ToT queries (engines differ; log when hit).
DEFAULT_MAX_QUERY_CHARS = 4000


def truncate_query(query: str, max_chars: int = DEFAULT_MAX_QUERY_CHARS) -> tuple[str, bool]:
    q = query.strip()
    if max_chars and len(q) > max_chars:
        return q[:max_chars], True
    return q, False


class Bm25Backend(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 1000) -> list[tuple[str, float]]:
        """Return top-k (docid, score) pairs."""

    def search_many(self, queries: Sequence[str], k: int = 1000) -> list[list[tuple[str, float]]]:
        return [self.search(q, k=k) for q in queries]


class PyseriniSearcher(Bm25Backend):
    """Wraps a Lucene index via pyserini (Anserini prebuilt for Tier B)."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        k1: float | None = None,
        b: float | None = None,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
    ):
        import os
        if not os.environ.get("JAVA_HOME"):
            candidates = [
                "/Library/Java/JavaVirtualMachines/temurin-25.jdk/Contents/Home",
                "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
            ]
            for cand in candidates:
                if Path(cand).exists():
                    os.environ["JAVA_HOME"] = cand
                    break

        from pyserini.search.lucene import LuceneSearcher

        self.index_dir = Path(index_dir)
        self.searcher = LuceneSearcher(str(self.index_dir))
        self.max_query_chars = max_query_chars
        self.truncated_events = 0
        if k1 is not None and b is not None:
            self.searcher.set_bm25(k1, b)
            self.k1, self.b = k1, b
        else:
            # Engine defaults; record whatever pyserini exposes if available
            self.k1, self.b = k1, b

    def search(self, query: str, k: int = 1000) -> list[tuple[str, float]]:
        q, truncated = truncate_query(query, self.max_query_chars)
        if truncated:
            self.truncated_events += 1
            logger.warning("BM25 query truncated to %s chars", self.max_query_chars)
        hits = self.searcher.search(q, k)
        return [(str(h.docid), float(h.score)) for h in hits]


class Bm25sSearcher(Bm25Backend):
    """In-memory bm25s index over a JSONL sample or sub-corpus."""

    def __init__(
        self,
        corpus_jsonl: str | Path | None = None,
        *,
        docs: list[dict[str, Any]] | None = None,
        lead_chars: int = 1000,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        stopwords: str | None = "en",
    ):
        import bm25s

        self.max_query_chars = max_query_chars
        self.truncated_events = 0
        self.lead_chars = lead_chars
        if docs is None:
            if corpus_jsonl is None:
                raise ValueError("Provide corpus_jsonl or docs")
            docs = []
            with Path(corpus_jsonl).open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        docs.append(json.loads(line))
        self.docids = [doc_id(d) for d in docs]
        corpus = [doc_text(d, lead_chars=lead_chars) for d in docs]
        self._bm25s = bm25s
        self.tokens = bm25s.tokenize(corpus, stopwords=stopwords)
        self.retriever = bm25s.BM25()
        self.retriever.index(self.tokens)

    def search(self, query: str, k: int = 1000) -> list[tuple[str, float]]:
        q, truncated = truncate_query(query, self.max_query_chars)
        if truncated:
            self.truncated_events += 1
        q_tokens = self._bm25s.tokenize([q], stopwords="en")
        k_eff = min(k, len(self.docids))
        results, scores = self.retriever.retrieve(q_tokens, k=k_eff)
        # results shape: (1, k) indices
        out: list[tuple[str, float]] = []
        for idx, score in zip(results[0], scores[0]):
            out.append((self.docids[int(idx)], float(score)))
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(path))
        meta = {"docids": self.docids, "lead_chars": self.lead_chars}
        Path(str(path) + ".docids.json").write_text(json.dumps(meta), encoding="utf-8")


def build_bm25_searcher(
    backend: str,
    *,
    index_dir: str | Path | None = None,
    corpus_jsonl: str | Path | None = None,
    k1: float | None = None,
    b: float | None = None,
    lead_chars: int = 1000,
) -> Bm25Backend:
    backend = backend.lower()
    if backend in {"pyserini", "anserini", "lucene"}:
        if index_dir is None:
            raise ValueError("index_dir required for pyserini backend")
        return PyseriniSearcher(index_dir, k1=k1, b=b)
    if backend in {"bm25s", "sample"}:
        return Bm25sSearcher(corpus_jsonl=corpus_jsonl, lead_chars=lead_chars)
    raise ValueError(f"Unknown BM25 backend: {backend}")


ANSERINI_INDEX_URL = (
    "https://files.webis.de/data-in-progress/trec-tot-2025-indices/"
    "trec-tot-2025-anserini-index.zip"
)
ANSERINI_INDEX_MD5 = "b04afdf33519013bf08857005a6cbd88"
