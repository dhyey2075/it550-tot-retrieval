#!/usr/bin/env python3
"""Stream TREC-ToT corpus gzip and keep only needed docids (for Tier-B dense text)."""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", type=Path, required=True, help="newline-separated docids")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--corpus-gz",
        type=Path,
        default=None,
        help="local corpus jsonl.gz; if missing, stream from Zenodo",
    )
    p.add_argument(
        "--url",
        default="https://zenodo.org/records/18235970/files/trec-tot-2025-corpus.jsonl.gz?download=1",
    )
    args = p.parse_args()

    want = {line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip ids already present in out
    already: set[str] = set()
    if args.out.exists() and args.out.stat().st_size > 0:
        with args.out.open(encoding="utf-8") as prev:
            for line in prev:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                already.add(str(obj.get("id") or obj.get("doc_id") or ""))
        want -= already
        print(f"[progress] resume: already={len(already)} still_need={len(want)}", flush=True)
    else:
        print(f"[progress] need {len(want)} docs", flush=True)

    if not want:
        print("[progress] nothing left to find", flush=True)
        return 0

    if args.corpus_gz and args.corpus_gz.exists() and args.corpus_gz.stat().st_size > 1_000_000:
        print(f"[progress] reading local {args.corpus_gz} ({args.corpus_gz.stat().st_size} bytes)", flush=True)
        opener = lambda: gzip.open(args.corpus_gz, "rt", encoding="utf-8")
    else:
        print(f"[progress] streaming {args.url}", flush=True)

        def opener():
            # large timeout; Zenodo streams are long
            resp = urllib.request.urlopen(args.url, timeout=600)
            return gzip.GzipFile(fileobj=resp)

    found = 0
    scanned = 0
    t0 = time.time()
    import io

    mode = "a" if already else "w"
    with opener() as gz, args.out.open(mode, encoding="utf-8") as out:
        # urlopen+GzipFile is binary; local gzip.open(..., "rt") is text
        if isinstance(gz, gzip.GzipFile):
            stream = io.TextIOWrapper(gz, encoding="utf-8")
        else:
            stream = gz
        for line in stream:
            scanned += 1
            if scanned % 50000 == 0:
                print(
                    f"  scanned={scanned} found={found}/{len(want)} "
                    f"elapsed={time.time()-t0:.0f}s out_mb={args.out.stat().st_size/1e6:.1f}",
                    flush=True,
                )
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            docid = str(obj.get("id") or obj.get("doc_id") or "")
            if docid in want:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                found += 1
                if found >= len(want):
                    break
    print(f"[progress] done found={found}/{len(want)} scanned={scanned} in {time.time()-t0:.0f}s", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0 if found else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        print(f"[fatal] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
