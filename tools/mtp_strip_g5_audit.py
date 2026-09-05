#!/usr/bin/env python3
"""Fleet audit for the glm5_next g5nsp MTP strip: every placed pack must
read flags=0, contain zero sentinel-layer (45) entries, carry a receipt
with a matching sha256, and sit chattr-locked."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
import subprocess
import sys

MAGIC = 0x33584C47
MTP_LAYER = 45
DIR_OFF = 512
ENTRY_BYTES = 64

patterns = [
    "/home/*/sparkdata/glm5_next.tp16/packs/*.g5nsp",
    "/home/*/sparkdata/glm5_next.tp4pp4/packs/*.g5nsp",
    "/home/*/sparkdata/glm5_next.tp8.fp8/packs/*.g5nsp",
    "/home/*/sparkdata/glm5_next.bf16.tp16/packs/*.g5nsp",
]

failures = 0
checked = 0
for pattern in patterns:
    for pack in sorted(glob.glob(pattern)):
        if pack.endswith((".receipt.json", ".compact.tmp")):
            continue
        checked += 1
        problems = []
        with open(pack, "rb") as f:
            raw = f.read(264)
        fields = struct.unpack_from("<20I", raw, 0)
        magic, version, flags = fields[0], fields[1], fields[5]
        count = fields[6]
        dir_off, file_bytes = struct.unpack_from("<QQ", raw, 80)
        if magic != MAGIC or version != 1:
            problems.append(f"magic/ver {magic:#x}/{version}")
        if flags != 0:
            problems.append(f"flags={flags}")
        if file_bytes != os.path.getsize(pack):
            problems.append("file_bytes mismatch")
        with open(pack, "rb") as f:
            f.seek(dir_off)
            raw_dir = f.read(count * ENTRY_BYTES)
        if len(raw_dir) != count * ENTRY_BYTES:
            problems.append("short directory")
        else:
            for i in range(count):
                block = raw_dir[i * ENTRY_BYTES:(i + 1) * ENTRY_BYTES]
                kind, layer = struct.unpack_from("<2I", block, 0)
                if layer == MTP_LAYER:
                    problems.append(f"MTP entry survives at {i} (kind={kind})")
                    break
        receipt_path = pack + ".receipt.json"
        digest = hashlib.sha256()
        with open(pack, "rb") as f:
            fd = f.fileno()
            while True:
                chunk = f.read(8 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
        sha = digest.hexdigest()
        try:
            receipt = json.load(open(receipt_path))
            if receipt.get("mtp") not in ("stripped", "none"):
                problems.append("receipt not marked")
            if receipt.get("output_sha256") != sha:
                problems.append("receipt sha mismatch")
        except Exception as error:
            problems.append(f"receipt unreadable: {error}")
        attrs = subprocess.run(["lsattr", pack], stdout=subprocess.PIPE,
                               text=True).stdout.split()
        if not attrs or "i" not in attrs[0]:
            problems.append("not chattr-locked")
        if problems:
            failures += 1
            print(f"FAIL {pack}: {'; '.join(problems)}")
        else:
            print(f"ok {pack} ({count} entries)")
print(f"AUDIT {'PASS' if failures == 0 else 'FAIL'}: {checked} packs, {failures} failures")
sys.exit(0 if failures == 0 else 1)
