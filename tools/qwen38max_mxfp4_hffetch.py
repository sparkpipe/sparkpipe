#!/usr/bin/env python3
"""Fetch the missing AMD Quark MXFP4 shards via huggingface_hub's Xet path.

Coexists with the curl-based downloader (tools/qwen38max_mxfp4_download.sh):
that loop takes files in list order from both ends of the split, so this
fetcher walks the MISSING set in REVERSE order - they meet in the middle.
Both writers produce only whole, size-verified files (atomic rename), so a
doubled fetch wastes bandwidth at worst, never corrupts.

Usage: qwen38max_mxfp4_hffetch.py --dest /mnt/model-warm/packbuild/qwen38max/amd-mxfp4
       [--workers N] [--shard-mod N/M] [--max-bytes N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = "amd/Qwen3.8-2.4T-A95B-Quark-MXFP4"


def file_list() -> list[tuple[str, int]]:
    with urllib.request.urlopen(
        f"https://huggingface.co/api/models/{REPO}?blobs=true", timeout=60) as r:
        d = json.load(r)
    files = [(s["rfilename"], int(s.get("size", 0)))
             for s in d.get("siblings", [])
             if not s["rfilename"].startswith(".")
             and s["rfilename"] not in ("README.md", "LICENSE")]
    return files


def fetch_one(name: str, size: int, dest: Path) -> bool:
    target = dest / name
    if target.exists() and target.stat().st_size == size:
        return False
    # hf_hub_download with local_dir writes the file into place atomically.
    from huggingface_hub import hf_hub_download
    hf_hub_download(REPO, name, local_dir=str(dest))
    if not target.exists() or target.stat().st_size != size:
        print(f"SIZE-MISMATCH {name}", file=sys.stderr)
        return False
    print(f"OK {name}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--shard-mod", default="0/1")
    parser.add_argument("--max-bytes", type=int, default=0, help="stop after moving this many bytes (0 = no cap)")
    args = parser.parse_args()
    n, m = (int(x) for x in args.shard_mod.split("/"))
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    files = file_list()
    mine = [(name, size) for index, (name, size) in enumerate(files) if index % m == n]
    missing = [(name, size) for name, size in reversed(mine)
               if not (dest / name).exists() or (dest / name).stat().st_size != size]
    total = sum(size for _, size in missing)
    print(f"{time.strftime('%FT%T')} hf-fetch start slice={args.shard_mod} "
          f"missing={len(missing)} bytes={total}", flush=True)
    moved = 0
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, name, size, dest): (name, size)
                   for name, size in missing}
        for future in futures:
            name, size = futures[future]
            try:
                if future.result():
                    moved += size
            except Exception as error:  # noqa: BLE001
                print(f"FAIL {name}: {error}", file=sys.stderr, flush=True)
            if args.max_bytes and moved >= args.max_bytes:
                print("byte cap reached", flush=True)
                break
    print(f"{time.strftime('%FT%T')} hf-fetch done moved={moved}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
