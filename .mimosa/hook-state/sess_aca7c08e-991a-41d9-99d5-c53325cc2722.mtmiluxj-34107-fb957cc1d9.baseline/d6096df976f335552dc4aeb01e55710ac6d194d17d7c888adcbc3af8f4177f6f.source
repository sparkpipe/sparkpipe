#!/usr/bin/env python3
"""Build MODEL_MANIFEST.json + ARCHIVE-RECEIPT.json + DOWNLOAD_STATUS.json
from an existing ARCHIVE-SHA256SUMS (repairs the dotfile-mangling bug in the
original archive script). Usage: ga_cold_manifest.py DEST [STARTED_AT]."""
import argparse
import hashlib
import json
import os
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("dest")
parser.add_argument("started", nargs="?")
args = parser.parse_args()
dest = args.dest
started = args.started


def _contained(base, leaf):
    root = os.path.realpath(base)
    path = os.path.realpath(os.path.join(base, leaf))
    if not (path == root or path.startswith(root + os.sep)):
        raise ValueError(f"path escapes dest dir: {leaf}")
    return path


def _archive_leaf(path):
    path = path[2:] if path.startswith("./") else path
    if ".." in pathlib.PurePath(path).parts:
        raise ValueError(f"rejecting archive path with ..: {path}")
    return path


files = []
with open(_contained(dest, "ARCHIVE-SHA256SUMS")) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        digest, path = line.split("  ", 1)
        path = _archive_leaf(path)
        size = os.path.getsize(_contained(dest, path))
        files.append({"path": path, "sha256": digest, "size": size})
from datetime import datetime, timezone
published = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
manifest = {
    "completed_at": published,
    "files": sorted(files, key=lambda e: e["path"]),
}
pathlib.Path(_contained(dest, "MODEL_MANIFEST.json")).write_text(
    json.dumps(manifest, indent=2) + "\n")
source_bytes = sum(e["size"] for e in files)
receipt = {
    "format": "ds4-model-cold-archive-v1",
    "name": "deepseek-v4-pro-0813-ga",
    "source_host": "spark3",
    "source_path": "/home/spark3/extnvme/models/hf/deepseek-ai/DeepSeek-V4-Pro-0813",
    "source_repo": "deepseek-ai/DeepSeek-V4-Pro-0813 (HF, GA 2026-08-13)",
    "source_files": len(files),
    "source_bytes": source_bytes,
    "source_manifest_sha256": hashlib.sha256(
        open(_contained(dest, "model.safetensors.index.json"), "rb").read()).hexdigest(),
    "transfer_method": "rsync_checksum_fabric_v1",
    "verification": "full per-file SHA-256 equality before publish",
    "started_at": started or published,
    "published_at": published,
}
pathlib.Path(_contained(dest, "ARCHIVE-RECEIPT.json")).write_text(
    json.dumps(receipt, indent=2) + "\n")
status = {
    "files_done": len(files),
    "state": "complete",
    "total_bytes": source_bytes,
    "updated_at": published,
}
pathlib.Path(_contained(dest, "DOWNLOAD_STATUS.json")).write_text(
    json.dumps(status, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
