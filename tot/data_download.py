"""Download and verify TREC-ToT 2025 query/qrel files (never Test)."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

# Prefer the record listed in scope.md. Same md5s appear on 15356599 (earlier version).
ZENODO_RECORD = "18235970"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# Official filenames -> local repo names under data/
# md5s from https://trec-tot.github.io/guidelines.html and Zenodo API
ALLOWED_DOWNLOADS = {
    "train-2025-queries.jsonl": {
        "local": "train-queries.jsonl",
        "md5": "288b7707b4e897f7447aac2cc2f613be",
        "n_expected": 143,
    },
    "train-2025-qrel.txt": {
        "local": "train-qrels.txt",
        "md5": "10a3c727fc5806ec4510f7a071b57cd7",
        "n_expected": 143,
    },
    "dev1-2025-queries.jsonl": {
        "local": "dev1-queries.jsonl",
        "md5": "b87c2f51d058de844e258a69b02e70fc",
        "n_expected": 142,
    },
    "dev1-2025-qrel.txt": {
        "local": "dev1-qrels.txt",
        "md5": "0c913ce8b5b287c73a6dfac662971e82",
        "n_expected": 142,
    },
    "dev3-2025-queries.jsonl": {
        "local": "dev3-queries.jsonl",
        "md5": "259c11645694a3c5230b66c7852d4d80",
        "n_expected": 536,
    },
    "dev3-2025-qrel.txt": {
        "local": "dev3-qrels.txt",
        "md5": "48ab0d24a5946861546e54064238477f",
        "n_expected": 536,
    },
}

BLOCKED = {
    "test-2025-queries.jsonl",
    "test-2025-qrel.txt",
    "test-2025-query-mapping.tsv",
}


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def ensure_downloads(data_dir: Path, *, force: bool = False) -> list[dict]:
    rows = []
    for remote_name, meta in ALLOWED_DOWNLOADS.items():
        if remote_name in BLOCKED:
            raise RuntimeError(f"Refusing blocked file: {remote_name}")
        local = data_dir / meta["local"]
        url = f"{ZENODO_BASE}/{remote_name}?download=1"
        status = "kept"
        if force or not local.exists() or md5_file(local) != meta["md5"]:
            download_file(url, local)
            status = "downloaded"
        digest = md5_file(local)
        ok = digest == meta["md5"]
        rows.append(
            {
                "remote": remote_name,
                "local": str(local),
                "url": url,
                "expected_md5": meta["md5"],
                "actual_md5": digest,
                "ok": ok,
                "n_expected": meta["n_expected"],
                "bytes": local.stat().st_size,
                "status": status,
                "zenodo_record": ZENODO_RECORD,
            }
        )
        if not ok:
            raise ValueError(f"md5 mismatch for {local}: got {digest}, expected {meta['md5']}")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download Train/Dev1/Dev3 only (never Test)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    rows = ensure_downloads(args.data_dir, force=args.force)
    for r in rows:
        print(f"{'OK' if r['ok'] else 'FAIL'} {r['local']} md5={r['actual_md5']} ({r['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
