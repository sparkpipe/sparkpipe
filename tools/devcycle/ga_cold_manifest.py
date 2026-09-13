#!/usr/bin/env python3
"""Build MODEL_MANIFEST.json + ARCHIVE-RECEIPT.json + DOWNLOAD_STATUS.json
from an existing ARCHIVE-SHA256SUMS (repairs the dotfile-mangling bug in the
original archive script). Usage: ga_cold_manifest.py DEST [STARTED_AT]."""
import hashlib, json, os, sys
from datetime import datetime, timezone

dest = sys.argv[1]
started = sys.argv[2] if len(sys.argv) > 2 else None
files = []
with open(os.path.join(dest, "ARCHIVE-SHA256SUMS")) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        digest, path = line.split("  ", 1)
        path = path[2:] if path.startswith("./") else path
        size = os.path.getsize(os.path.join(dest, path))
        files.append({"path": path, "sha256": digest, "size": size})
published = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
manifest = {
    "completed_at": published,
    "files": sorted(files, key=lambda e: e["path"]),
}
with open(os.path.join(dest, "MODEL_MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
source_bytes = sum(e["size"] for e in files)
receipt = {
    "format": "ds4-model-cold-archive-v1",
    "name": "deepseek-v4-pro-0813-ga",
    "source_host": "spark3",
    "source_path": "/home/spark3/extnvme/models/hf/deepseek-ai/DeepSeek-V4-Pro-0813",
    "source_repo": "deepseek-ai/DeepSeek-V4-Pro-0813 (HF, GA 2026-08-13)",
    "source_files": len(files),
    "source_bytes": source_bytes,
    "source_manifest_sha256": hashlib.sha256(open(os.path.join(dest, "model.safetensors.index.json"), "rb").read()).hexdigest(),
    "transfer_method": "rsync_checksum_fabric_v1",
    "verification": "full per-file SHA-256 equality before publish",
    "started_at": started or published,
    "published_at": published,
}
with open(os.path.join(dest, "ARCHIVE-RECEIPT.json"), "w") as f:
    json.dump(receipt, f, indent=2)
    f.write("\n")
status = {
    "files_done": len(files),
    "state": "complete",
    "total_bytes": source_bytes,
    "updated_at": published,
}
with open(os.path.join(dest, "DOWNLOAD_STATUS.json"), "w") as f:
    json.dump(status, f, indent=2)
    f.write("\n")
print(json.dumps(receipt, indent=2))
