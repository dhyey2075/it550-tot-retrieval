"""Fairness checklist (plan_v2/03 §7) — verified by code."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.io_utils import DEPTH, load_qrels, md5_file


@dataclass
class RunMeta:
    run_path: Path
    exp_id: str
    split: str
    corpus_tier: str
    config_hash: str | None = None
    prompt_version: str | None = None
    retrieve_depth: int = DEPTH
    fuse_out_depth: int = DEPTH
    qrels_path: Path | None = None


@dataclass
class FairnessReport:
    ok: bool
    messages: list[str] = field(default_factory=list)

    def print(self) -> None:
        status = "PASS" if self.ok else "FAIL"
        print(f"assert_fair: {status}")
        for m in self.messages:
            print(f"  - {m}")


def _load_sidecar(run_path: Path) -> dict[str, Any]:
    side = run_path.with_suffix(".json")
    if not side.exists():
        return {}
    return json.loads(side.read_text(encoding="utf-8"))


def assert_fair(
    runs: list[RunMeta],
    *,
    expected_qids: set[str] | None = None,
    frozen_prompt_version: str | None = None,
    require_same_tier: bool = True,
) -> FairnessReport:
    """
    Check that runs are comparable. Prints and returns a FairnessReport.

    Does not raise by default — caller decides whether to abort.
    """
    msgs: list[str] = []
    ok = True

    if len(runs) < 1:
        return FairnessReport(ok=False, messages=["no runs provided"])

    # qrels hash
    qrels_hashes = []
    for r in runs:
        if r.qrels_path and r.qrels_path.exists():
            qrels_hashes.append(md5_file(r.qrels_path))
        else:
            msgs.append(f"WARNING {r.run_path.name}: no qrels_path for hash check")
    if qrels_hashes and len(set(qrels_hashes)) > 1:
        ok = False
        msgs.append(f"FAIL: qrels md5 mismatch across runs: {set(qrels_hashes)}")
    elif qrels_hashes:
        msgs.append(f"OK: same qrels md5={qrels_hashes[0]}")

    # corpus tier
    tiers = {r.corpus_tier for r in runs}
    if require_same_tier and len(tiers) > 1:
        ok = False
        msgs.append(f"FAIL: mixed corpus_tier labels: {tiers}")
    elif len(tiers) > 1:
        msgs.append(f"WARNING: mixed corpus_tier labels: {tiers}")
    else:
        msgs.append(f"OK: corpus_tier={next(iter(tiers))}")

    # depths
    for r in runs:
        if r.retrieve_depth != DEPTH or r.fuse_out_depth != DEPTH:
            msgs.append(
                f"WARNING {r.run_path.name}: depths retrieve={r.retrieve_depth} "
                f"fuse_out={r.fuse_out_depth} (project default {DEPTH})"
            )
    msgs.append(f"OK: expected retrieve/fuse_out depth={DEPTH}")

    # config hash present
    for r in runs:
        ch = r.config_hash
        if not ch:
            side = _load_sidecar(r.run_path)
            ch = None
            if side:
                cfg = side.get("config") or {}
                ch = side.get("config_hash") or cfg.get("hash")
        if not ch:
            msgs.append(f"WARNING {r.run_path.name}: config_hash missing")
        else:
            msgs.append(f"OK {r.run_path.name}: config_hash={ch}")

    # prompt versions for LLM exps
    llm_exps = {"E4", "E5", "E6", "E7", "E8", "E9", "E4-R", "E5-R"}
    for r in runs:
        if r.exp_id.split("-")[0] in {e.split("-")[0] for e in llm_exps} or r.exp_id in llm_exps:
            pv = r.prompt_version
            if frozen_prompt_version and pv and pv != frozen_prompt_version:
                ok = False
                msgs.append(
                    f"FAIL {r.run_path.name}: prompt_version={pv} != frozen={frozen_prompt_version}"
                )
            elif frozen_prompt_version and not pv:
                msgs.append(f"WARNING {r.run_path.name}: LLM-ish exp missing prompt_version")
            elif pv:
                msgs.append(f"OK {r.run_path.name}: prompt_version={pv}")

    if expected_qids is not None:
        from evaluation.io_utils import load_run

        for r in runs:
            if not r.run_path.exists():
                msgs.append(f"WARNING {r.run_path}: run file missing; skip qid coverage check")
                continue
            ranking = load_run(r.run_path, warn_duplicates=False)
            run_qids = set(ranking.keys())
            missing = expected_qids - run_qids
            extra = run_qids - expected_qids
            if missing:
                msgs.append(
                    f"INFO {r.run_path.name}: {len(missing)} qrels qids missing from run "
                    "(scored as 0 by evaluator)"
                )
            if extra:
                msgs.append(f"WARNING {r.run_path.name}: {len(extra)} run qids not in qrels")
        msgs.append(f"OK: evaluation set size expected={len(expected_qids)}")

    report = FairnessReport(ok=ok, messages=msgs)
    report.print()
    return report
