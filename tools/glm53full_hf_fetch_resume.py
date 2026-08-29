#!/usr/bin/env python3
"""glm53full lane: resume a stalled Hugging Face staging fetch, house receipts.

The two 5.3-full official sources (zai-org/GLM-5.3-BF16 @304b8051cfb2,
zai-org/GLM-5.3 FP8 @935644c05e76) were fetched by an ad-hoc process that
died during the 2026-08-29 ceph incident; their .staging dirs sat static
for ~12h with no owning process (lsof clean). This tool resumes IN PLACE:

  - files already present at the exact expected byte size are kept,
  - *.ds4tmp partials are Range-resumed (206 or restart),
  - every file is then sha256-verified against the Hub LFS blob sha
    (non-LFS files: exact size check, same as ds4_hf_publish.py),
  - a ds4-hf-warm-download-receipt-v1 + PUBLISHED marker is written and
    the stage dir is atomically promoted to the warm-proper destination
    -- the identical end-state contract as ~/ds4_hf_publish.py.

No quantization: this moves publisher bytes verbatim (quant policy).

Usage (run on the node holding the staging dir, from a git checkout):
  python3 tools/glm53full_hf_fetch_resume.py \
      --repo zai-org/GLM-5.3 --revision 935644c0... \
      --stage /mnt/model-warm/.staging/glm-5.3-935644c0 \
      --destination glm-5.3-fp8 --license-class zai-commercial-use-granted
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def hub_manifest(repo: str, revision: str) -> tuple[str, list[dict]]:
    url = f"{HF_API}/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=120) as response:
        info = json.load(response)
    if info.get("sha") != revision:
        raise RuntimeError(f"Hub resolved {revision} to unexpected commit {info.get('sha')}")
    expected = []
    for sibling in info.get("siblings", []):
        lfs = sibling.get("lfs") or {}
        # python-lib shape is lfs.oid ("sha256:<hex>"); REST ?blobs=true
        # shape is lfs.sha256 ("<hex>"). Accept both.
        lfs_sha = (lfs.get("oid") or "").removeprefix("sha256:") or lfs.get("sha256") or None
        expected.append(
            {
                "path": sibling["rfilename"],
                "expected_bytes": int(sibling.get("size") or 0),
                "expected_sha256": lfs_sha,
            }
        )
    return info["sha"], expected


def fetch_one(stage: Path, base: dict, attempts: int = 4) -> dict:
    """Download one file (skipping an exact-size hit; Range-resuming .ds4tmp)."""
    relative = base["path"]
    final = stage / relative
    if final.is_file() and final.stat().st_size == base["expected_bytes"]:
        return {"path": relative, "action": "already_complete"}
    partial = stage / (relative + ".ds4tmp")
    partial.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    expected = base["expected_bytes"]
    for attempt in range(attempts):
        resume_from = partial.stat().st_size if partial.is_file() else 0
        request = urllib.request.Request(
            HF_RESOLVE.format(repo=base["repo"], revision=base["revision"], path=relative)
        )
        request.add_header("User-Agent", "sparkpipe-glm53full-fetch/1.0")
        if resume_from:
            request.add_header("Range", f"bytes={resume_from}-")
        mode = "ab" if resume_from else "wb"
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                code = getattr(response, "status", 200)
                if resume_from and code != 206:
                    # server ignored the range: drop the partial, retry from zero
                    partial.unlink(missing_ok=True)
                    resume_from = 0
                    mode = "wb"
                    continue
                buf = bytearray(8 * 1024 * 1024)
                with open(partial, mode) as stream:
                    while True:
                        read = response.readinto(buf)
                        if not read:
                            break
                        stream.write(buf[:read])
            got = partial.stat().st_size
            want = expected if mode == "wb" else resume_from + expected
            if got != want:
                raise RuntimeError(f"size mismatch mid-stream: have {got} want {want}")
            os.replace(partial, final)
            return {"path": relative, "action": "resumed" if mode == "ab" else "downloaded"}
        except Exception as error:  # noqa: BLE001 - retry with backoff
            last_error = error
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"fetch failed for {relative}: {last_error}")


def verify_one(stage: Path, item: dict) -> dict:
    path = stage / item["path"]
    if not path.is_file():
        raise RuntimeError(f"missing file: {item['path']}")
    actual_bytes = path.stat().st_size
    expected_bytes = item["expected_bytes"]
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise RuntimeError(
            f"size mismatch: {item['path']} expected={expected_bytes} actual={actual_bytes}"
        )
    actual_sha256 = None
    expected_sha256 = item["expected_sha256"]
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with path.open("rb", buffering=8 * 1024 * 1024) as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"sha256 mismatch: {item['path']} expected={expected_sha256} actual={actual_sha256}"
            )
    return {
        "path": item["path"],
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "hub_sha256": expected_sha256,
    }


def find_partial_files(stage: Path) -> list[str]:
    suffixes = (".part", ".incomplete", ".ds4tmp", ".lock")
    return sorted(
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(stage).parts
        and path.name.endswith(suffixes)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--stage", required=True, help="existing staging dir to resume")
    parser.add_argument("--destination", required=True, help="warm-proper dir name")
    parser.add_argument("--license-class", required=True)
    parser.add_argument("--warm-root", default="/mnt/model-warm")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify-workers", type=int, default=6)
    args = parser.parse_args()

    warm_root = Path(args.warm_root)
    stage = Path(args.stage)
    final = warm_root / args.destination
    status = warm_root / ".staging" / f"{args.destination}.status.json"
    log_base = {
        "format": "ds4-hf-warm-publish-status-v1",
        "repo": args.repo,
        "revision": args.revision,
        "destination": str(final),
        "stage": str(stage),
        "license_class": args.license_class,
        "pid": os.getpid(),
        "resumed_by": "tools/glm53full_hf_fetch_resume.py",
    }
    if (final / "PUBLISHED").is_file():
        marker = json.loads((final / "PUBLISHED").read_text(encoding="utf-8"))
        if marker.get("revision") != args.revision:
            raise RuntimeError(f"published destination has another revision: {final}")
        atomic_json(status, {**log_base, "phase": "already_published", "updated_at": utc_now()})
        print(f"already_published repo={args.repo} destination={final}")
        return 0
    if final.exists():
        raise RuntimeError(f"unpublished destination already exists: {final}")
    if not stage.is_dir():
        raise RuntimeError(f"staging dir does not exist: {stage}")

    revision_sha, expected = hub_manifest(args.repo, args.revision)
    total_bytes = sum(item["expected_bytes"] for item in expected)
    started = utc_now()
    atomic_json(
        status,
        {
            **log_base,
            "phase": "downloading",
            "started_at": started,
            "updated_at": started,
            "expected_files": len(expected),
            "expected_bytes": total_bytes,
        },
    )
    print(f"resuming {stage} repo={args.repo} files={len(expected)} bytes={total_bytes}", flush=True)

    bases = [dict(item, repo=args.repo, revision=args.revision) for item in expected]
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_one, stage, base): base for base in bases}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            done += 1
            if done % 10 == 0 or done == len(bases):
                print(f"fetch {done}/{len(bases)}: {result['path']} {result['action']}", flush=True)

    partials = find_partial_files(stage)
    if partials:
        raise RuntimeError(f"partial files remain: {partials[:8]}")

    atomic_json(
        status,
        {
            **log_base,
            "phase": "verifying",
            "started_at": started,
            "updated_at": utc_now(),
            "expected_files": len(expected),
            "expected_bytes": total_bytes,
        },
    )
    verified: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.verify_workers) as executor:
        futures = {executor.submit(verify_one, stage, item): item for item in expected}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            verified.append(future.result())
            if index % 25 == 0 or index == len(expected):
                atomic_json(
                    status,
                    {
                        **log_base,
                        "phase": "verifying",
                        "started_at": started,
                        "updated_at": utc_now(),
                        "verified_files": index,
                        "expected_files": len(expected),
                        "verified_bytes": sum(item["bytes"] for item in verified),
                    },
                )
    verified.sort(key=lambda item: item["path"])
    receipt = {
        "format": "ds4-hf-warm-download-receipt-v1",
        "repo": args.repo,
        "revision": args.revision,
        "license_class": args.license_class,
        "started_at": started,
        "verified_at": utc_now(),
        "files": len(verified),
        "bytes": sum(item["bytes"] for item in verified),
        "file_records": verified,
    }
    atomic_json(stage / "DOWNLOAD-RECEIPT.json", receipt)
    marker = {
        "format": "ds4-hf-warm-published-v1",
        "repo": args.repo,
        "revision": args.revision,
        "license_class": args.license_class,
        "published_at": utc_now(),
        "receipt": "DOWNLOAD-RECEIPT.json",
    }
    atomic_json(stage / "PUBLISHED", marker)
    os.replace(stage, final)
    atomic_json(status, {**log_base, "phase": "published", "updated_at": utc_now()})
    print(
        f"published repo={args.repo} revision={args.revision} destination={final} bytes={receipt['bytes']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAILED {type(error).__name__}: {error}", file=sys.stderr)
        raise
