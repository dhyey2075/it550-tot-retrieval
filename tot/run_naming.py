"""TREC run naming and sidecar helpers (plan_v2/01 §6.5)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def make_run_name(
    exp_id: str,
    split: str,
    corpus_tier: str,
    prompt_ver: str,
    date: str | None = None,
) -> str:
    """
    Pattern: E{n}_{split}_{corpusTier}_{promptVer}_{YYYYMMDD}
    Example: E6_dev3_tierA_p3_20260301
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
    parts = [exp_id, split, corpus_tier, prompt_ver, date]
    for p in parts:
        if not p or any(c.isspace() for c in p):
            raise ValueError(f"Invalid run-name part: {p!r}")
    return "_".join(parts)


def write_run_sidecar(
    run_path: str | Path,
    *,
    config_snapshot: dict[str, Any],
    git_commit: str | None,
    wall_time_sec: float | None,
    n_queries: int,
    docs_per_query: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_path = Path(run_path)
    sidecar = run_path.with_suffix(run_path.suffix + ".json") if run_path.suffix else Path(str(run_path) + ".json")
    if run_path.suffix == ".run":
        sidecar = run_path.with_suffix(".json")
    payload = {
        "run_file": run_path.name,
        "config": config_snapshot,
        "git_commit": git_commit,
        "wall_time_sec": wall_time_sec,
        "n_queries": n_queries,
        "docs_per_query": docs_per_query,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload["extra"] = extra
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sidecar
