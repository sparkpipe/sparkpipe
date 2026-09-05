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
    # canonical arms (the 2026-09-05 rename); packs may sit at the arm root
    # (legacy spark0 layout) or under packs/, named either the legacy
    # glm5_next_stage.* form or the canonical <arm>.rank<h>.sp form
    "/home/*/sparkdata/glm53flash.bf16.tp16/*",
    "/home/*/sparkdata/glm53flash.bf16.tp16/packs/*",
    "/home/*/sparkdata/glm53flash.fp8.tp4pp4/*",
    "/home/*/sparkdata/glm53flash.fp8.tp4pp4/packs/*",
    "/home/*/sparkdata/glm53flash.fp8.tp8/*",
    "/home/*/sparkdata/glm53flash.fp8.tp8/packs/*",
    "/home/*/sparkdata/glm53flash.fp8.tp16/packs/*",
    "/home/*/sparkdata/glm53flash.nvfp4.tp16/packs/*",
]

# real packs (both naming generations) start with the family prefixes;
# arm-dir debris (residentd.log.rankN-flake-*, logs, json) does not
PACK_PREFIXES = ("glm53flash.", "glm5_next_stage.")
SKIP_SUFFIXES = (".receipt.json", ".packer-receipt.json", ".compact.tmp",
                 ".sha256", ".ck128", ".experts", ".mtp.receipt",
                 ".symlinkfix.receipt")

failures = 0
checked = 0
seen = set()
sys.stdout.reconfigure(line_buffering=True)
for pattern in patterns:
    for pack in sorted(glob.glob(pattern)):
        if not os.path.isfile(pack) or pack in seen:
            continue
        seen.add(pack)
        base = os.path.basename(pack)
        if not base.startswith(PACK_PREFIXES):
            continue
        if pack.endswith(SKIP_SUFFIXES):
            continue
        checked += 1
        problems = []
        try:
            with open(pack, "rb") as f:
                raw = f.read(264)
            fields = struct.unpack_from("<20I", raw, 0)
            magic, version, flags = fields[0], fields[1], fields[5]
            count = fields[6]
            dir_off, file_bytes = struct.unpack_from("<QQ", raw, 80)
        except Exception as error:
            failures += 1
            print(f"FAIL {pack}: header unreadable: {error}")
            continue
        if magic != MAGIC or version != 1:
            problems.append(f"magic/ver {magic:#x}/{version}")
        if flags != 0:
            problems.append(f"flags={flags}")
        if file_bytes != os.path.getsize(pack):
            problems.append("file_bytes mismatch")
        try:
            with open(pack, "rb") as f:
                f.seek(dir_off)
                raw_dir = f.read(count * ENTRY_BYTES)
        except Exception as error:
            problems.append(f"directory unreadable at {dir_off}: {error}")
            raw_dir = b""
        if len(raw_dir) != count * ENTRY_BYTES:
            if "directory unreadable" not in " ".join(problems):
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
        receipt = None
        for candidate in (receipt_path, pack + ".g5nsp.receipt.json"):
            try:
                receipt = json.load(open(candidate))
                break
            except Exception:
                continue
        if receipt is None:
            problems.append(f"receipt unreadable: no .receipt.json or "
                            f".g5nsp.receipt.json beside pack")
        else:
            if receipt.get("mtp") not in ("stripped", "none"):
                problems.append("receipt not marked")
            if receipt.get("output_sha256") != sha:
                problems.append("receipt sha mismatch")
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
