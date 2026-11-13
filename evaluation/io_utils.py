"""Load TREC run files and qrels; normalize docids."""

from __future__ import annotations

import hashlib
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DEPTH = 1000


def normalize_docid(docid: str | int) -> str:
    return str(docid).strip()


def normalize_qid(qid: str | int) -> str:
    return str(qid).strip()


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_qrels(path: str | Path) -> dict[str, str]:
    """
    Load single-relevant qrels: {qid: target_docid}.

    Expects `qid 0 docid rel`. Raises if a qid has more than one judgment
    or if rel is not positive.
    """
    path = Path(path)
    targets: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_no}: expected 4 columns, got {parts!r}")
            qid, _iter, docid, rel_s = parts
            qid = normalize_qid(qid)
            docid = normalize_docid(docid)
            rel = int(rel_s)
            if rel <= 0:
                continue
            if qid in targets:
                raise ValueError(f"{path}:{line_no}: multiple relevant docs for qid={qid}")
            targets[qid] = docid
    return targets


def load_run(
    path: str | Path,
    *,
    depth: int = DEPTH,
    warn_duplicates: bool = True,
) -> dict[str, list[str]]:
    """
    Load a TREC run into {qid: [docid, ...]} sorted by score descending.

    Does not trust the rank column. Truncates each list to `depth`.
    Duplicate docids within a query: keep first (after score sort), warn.
    """
    path = Path(path)
    # qid -> list of (score, docid, line_no) then sort
    raw: dict[str, list[tuple[float, str, int]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no}: expected >=6 TREC columns, got {parts!r}")
            qid, _q0, docid, _rank, score_s, _run = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            qid = normalize_qid(qid)
            docid = normalize_docid(docid)
            score = float(score_s)
            raw[qid].append((score, docid, line_no))

    rankings: dict[str, list[str]] = {}
    for qid, rows in raw.items():
        # Stable tie-break: higher score first, then earlier line
        rows.sort(key=lambda t: (-t[0], t[2]))
        seen: set[str] = set()
        ordered: list[str] = []
        dup_count = 0
        for _score, docid, _ln in rows:
            if docid in seen:
                dup_count += 1
                continue
            seen.add(docid)
            ordered.append(docid)
            if len(ordered) >= depth:
                break
        if dup_count and warn_duplicates:
            warnings.warn(
                f"run {path.name} qid={qid}: dropped {dup_count} duplicate docid(s); kept first",
                stacklevel=2,
            )
        rankings[qid] = ordered
    return rankings


def write_trec_run(
    path: str | Path,
    rankings: dict[str, Iterable[str]],
    *,
    run_name: str,
    scores: dict[str, Iterable[float]] | None = None,
) -> None:
    """Write a TREC run file. ranks are 1-based; scores default to descending ranks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(rankings.keys(), key=lambda x: (len(x), x)):
            docs = list(rankings[qid])
            sc = list(scores[qid]) if scores and qid in scores else None
            for i, docid in enumerate(docs, start=1):
                score = float(sc[i - 1]) if sc is not None else float(len(docs) - i + 1)
                f.write(f"{qid} Q0 {docid} {i} {score:.8f} {run_name}\n")
