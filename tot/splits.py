"""Split loading guard — Test is never allowed; Dev2 only if amendment A1 is approved."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Default allowed splits. Add "dev2" only after D-10 freezes amendment A1 = yes.
ALLOWED_SPLITS = frozenset({"train", "dev1", "dev3"})

# Filenames expected under data/ once plan 02 downloads complete.
SPLIT_FILES = {
    "train": {"queries": "train-queries.jsonl", "qrels": "train-qrels.txt"},
    "dev1": {"queries": "dev1-queries.jsonl", "qrels": "dev1-qrels.txt"},
    "dev2": {"queries": "dev2-queries.jsonl", "qrels": "dev2-qrels.txt"},
    "dev3": {"queries": "dev3-queries.jsonl", "qrels": "dev3-qrels.txt"},
}


def assert_split_allowed(
    name: str,
    *,
    allow_dev2: bool = False,
    allowed: Iterable[str] | None = None,
) -> str:
    """Normalize and validate a split name. Raises ValueError for blocked splits."""
    key = name.strip().lower()
    if allowed is not None:
        allowed_set = frozenset(s.strip().lower() for s in allowed)
    else:
        allowed_set = set(ALLOWED_SPLITS)
        if allow_dev2:
            allowed_set.add("dev2")
        allowed_set = frozenset(allowed_set)

    if key == "test" or key not in allowed_set:
        raise ValueError(
            f"Split '{name}' is not allowed in this project. "
            f"Allowed: {sorted(allowed_set)}. "
            "Test is never loaded; Dev2 requires amendment A1 (D-10)."
        )
    return key


def load_split(
    name: str,
    data_dir: str | Path = "data",
    *,
    allow_dev2: bool = False,
) -> dict[str, Path]:
    """
    Resolve query/qrel paths for an allowed split.

    Does not download files; only checks the split is permitted and returns paths.
    Callers must verify the files exist before reading.
    """
    key = assert_split_allowed(name, allow_dev2=allow_dev2)
    root = Path(data_dir)
    files = SPLIT_FILES[key]
    return {
        "split": key,
        "queries": root / files["queries"],
        "qrels": root / files["qrels"],
    }
