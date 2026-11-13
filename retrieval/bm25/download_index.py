"""Download and unpack the official Anserini TREC-ToT 2025 index."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

from retrieval.bm25.search import ANSERINI_INDEX_MD5, ANSERINI_INDEX_URL


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_anserini_index(
    dest_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "trec-tot-2025-anserini-index.zip"
    # After unzip, look for a directory that looks like a Lucene index
    marker = dest_dir / "INDEX_READY"

    if marker.exists() and not force:
        index_root = Path(marker.read_text(encoding="utf-8").strip())
        if index_root.exists():
            print(f"Reusing index at {index_root}")
            return index_root

    if force or not zip_path.exists() or md5_file(zip_path) != ANSERINI_INDEX_MD5:
        print(f"Downloading {ANSERINI_INDEX_URL} -> {zip_path}")
        with urlopen(ANSERINI_INDEX_URL) as resp, zip_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
        digest = md5_file(zip_path)
        if digest != ANSERINI_INDEX_MD5:
            raise ValueError(f"md5 mismatch: got {digest}, expected {ANSERINI_INDEX_MD5}")
        print("md5 OK")

    print(f"Unpacking {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # Heuristic: find directory containing segments* or lucene index files
    candidates = [
        p
        for p in dest_dir.rglob("*")
        if p.is_dir() and (any(p.glob("segments*")) or (p / "write.lock").exists())
    ]
    if not candidates:
        candidates = [p for p in dest_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"Could not locate Lucene index under {dest_dir}")
    index_root = max(candidates, key=lambda p: sum(1 for _ in p.glob("*")))
    marker.write_text(str(index_root.resolve()) + "\n", encoding="utf-8")
    print(f"Index ready: {index_root}")
    return index_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=Path("indices/anserini"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    download_anserini_index(args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
