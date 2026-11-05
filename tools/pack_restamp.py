from __future__ import annotations

import datetime
import hashlib
import json
import os


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    offset = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 24)
            if not block:
                return digest.hexdigest()
            digest.update(block)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(handle.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
            offset += len(block)


def sidecar_digest(pack: str) -> str:
    with open(pack + ".sha256") as handle:
        return handle.read().split()[0]


def restamp(pack: str, header: bytes, offset: int, old: bytes, new: bytes, label: str, report: dict) -> dict:
    if header[offset:offset + len(old)] != old or len(old) != len(new):
        raise SystemExit(f"{pack}: header bytes at {offset} are not the expected source value")
    recorded = sidecar_digest(pack)
    before = file_sha256(pack)
    if before != recorded:
        raise SystemExit(f"{pack}: content sha256 {before} != sidecar {recorded}; refusing to restamp")
    backup = f"{pack}.header-{label}"
    with open(backup, "wb") as handle:
        handle.write(header)
    with open(pack, "r+b") as handle:
        handle.seek(offset)
        handle.write(new)
        handle.flush()
        os.fsync(handle.fileno())
    with open(pack, "rb") as handle:
        handle.seek(offset)
        if handle.read(len(new)) != new:
            raise SystemExit(f"{pack}: stamp did not land")
    after = file_sha256(pack)
    with open(pack + ".sha256.new", "w") as handle:
        handle.write(f"{after}  {os.path.basename(pack)}\n")
    os.replace(pack + ".sha256.new", pack + ".sha256")
    receipt = dict(report, action="restamped", sha256_before=before, sha256_after=after, header_backup=backup,
                   utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(pack + ".restamp.json", "w") as handle:
        json.dump(receipt, handle, indent=1)
        handle.write("\n")
    return receipt
