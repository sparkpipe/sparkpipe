#!/usr/bin/env python3
"""Fetch the OFFICIAL Qwen/Qwen3.8-Flash-Next-FP8 release into a warm dir.

Policy (operator directive 2026-08-28, commit 8e0aca7): no self-made
quantizations - official publisher releases only; packers repackage. This
archive is the publisher's own fine-grained fp8 (weight_block_size
[128,128]); the lane packer consumes it WITHOUT requantizing.

Downloads every repo file at the pinned commit, verifying each against
the HF LFS sha256 (provenance.json is written first, so the fetch is
auditable even mid-flight). Idempotent: already-correct files are
skipped, mismatches are re-fetched.

  python3 tools/qwen4_flash_fetch_fp8.py --dest /mnt/model-warm/qwen3.8-flash-next-fp8 \
      [--commit 970c569adaca6b35532111fd6b27351b2baefe50] [--workers 6]
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request

REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
DEFAULT_COMMIT = "970c569adaca6b35532111fd6b27351b2baefe50"


def api_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def list_tree(commit):
    entries = []
    cursor = f"https://huggingface.co/api/models/{REPO}/tree/{commit}?recursive=true"
    while cursor:
        with urllib.request.urlopen(cursor, timeout=60) as response:
            page = json.load(response)
            link = response.headers.get("Link", "")
        entries.extend(page)
        cursor = None
        if 'rel="next"' in link:
            cursor = link.split("<", 1)[1].split(">", 1)[0]
    return entries


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_one(entry, commit, dest, attempts=4):
    path = entry["path"]
    target = os.path.join(dest, path)
    expected = (entry.get("lfs") or {}).get("oid")
    size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
    if os.path.isfile(target) and size and os.path.getsize(target) == size:
        if not expected or sha256_file(target) == expected:
            return path, size, "present"
    os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/{commit}/{path}"
    for attempt in range(attempts):
        tmp = target + ".part"
        try:
            started = time.time()
            urllib.request.urlretrieve(url, tmp)
            if size and os.path.getsize(tmp) != size:
                raise IOError(f"short read {os.path.getsize(tmp)} != {size}")
            got = sha256_file(tmp) if expected else None
            if expected and got != expected:
                raise IOError(f"sha mismatch {got} != {expected}")
            os.replace(tmp, target)
            rate = size / max(time.time() - started, 0.001) / (1 << 20)
            return path, size, f"{rate:.0f}MiB/s"
        except Exception as error:  # noqa: BLE001 - report and retry
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt + 1 == attempts:
                raise
            print(f"retry {path}: {error}", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    entries = [e for e in list_tree(args.commit) if e["type"] == "file"]
    total = sum((e.get("size") or (e.get("lfs") or {}).get("size") or 0) for e in entries)
    provenance = {
        "repo": REPO,
        "commit": args.commit,
        "files": [
            {"path": e["path"], "size": (e.get("size") or (e.get("lfs") or {}).get("size") or 0),
             "sha256": (e.get("lfs") or {}).get("oid")}
            for e in entries
        ],
    }
    with open(os.path.join(args.dest, "provenance.json"), "w") as handle:
        json.dump(provenance, handle, indent=1, sort_keys=True)
    print(f"q4fp8_fetch commit={args.commit} files={len(entries)} total_gib={total / 2**30:.2f}", flush=True)

    done_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, e, args.commit, args.dest): e for e in entries}
        for future in concurrent.futures.as_completed(futures):
            path, size, note = future.result()
            done_bytes += size
            print(f"fetched {path} {size} ({note}) total={done_bytes / 2**30:.1f}GiB", flush=True)
    print("Q4FP8_FETCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
