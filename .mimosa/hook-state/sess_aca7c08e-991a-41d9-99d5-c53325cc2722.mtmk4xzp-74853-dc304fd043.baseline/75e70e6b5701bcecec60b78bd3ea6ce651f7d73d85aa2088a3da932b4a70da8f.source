#!/usr/bin/env python3
"""Fleet audit for the flash-family MTP strip: every placed pack must
read mtp_layer_count=0, contain zero MTP entries, carry an updated
receipt (mtp: stripped + a sha256), and sit chattr-locked."""

from __future__ import annotations

import glob
import json
import struct
import subprocess
import sys

MTP_LAYER = 0xFFFFFFFE
GLOBAL_KINDS = {28, 29, 30, 31, 43, 44}

patterns = [
    "/home/*/sparkdata/qwenflash.tp8.fp8/packs/*.spstage",
    "/home/*/sparkdata/qwenflash.tp4pp4.fp8/packs/*.spstage",
    "/home/*/sparkdata/qwen4_flash.tp4/packs_v4/*.qwen4_flashsp",
    "/home/*/qf_rank4/*.pack",
]

failures = 0
checked = 0
for pattern in patterns:
    for pack in sorted(glob.glob(pattern)):
        if pack.endswith(".receipt.json") or pack.endswith(".compact.tmp"):
            continue
        checked += 1
        problems = []
        with open(pack, "rb") as f:
            raw = f.read(120)
        u32s = struct.unpack_from("<26I", raw, 0)
        count = u32s[4]
        mtp_field = u32s[25]
        dir_off, file_bytes = struct.unpack_from("<2Q", raw, 104)
        if mtp_field != 0:
            problems.append(f"mtp_layer_count={mtp_field}")
        if file_bytes != __import__("os").path.getsize(pack):
            problems.append("file_bytes mismatch")
        with open(pack, "rb") as f:
            f.seek(dir_off)
            raw_dir = f.read(count * 56)
        if len(raw_dir) != count * 56:
            problems.append("short directory")
        else:
            for i in range(count):
                block = raw_dir[i * 56:(i + 1) * 56]
                kind, layer = struct.unpack_from("<2I", block, 0)
                if layer == MTP_LAYER or (layer == 0xFFFFFFFF and kind in GLOBAL_KINDS):
                    problems.append(f"MTP entry survives at {i} (kind={kind})")
                    break
        receipt_path = pack + ".receipt.json"
        try:
            receipt = json.load(open(receipt_path))
            if receipt.get("mtp") not in ("stripped", "none"):
                problems.append("receipt not marked")
            if not receipt.get("output_sha256"):
                problems.append("receipt has no sha256")
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
            print(f"ok {pack} ({count} tensors)")
print(f"AUDIT {'PASS' if failures == 0 else 'FAIL'}: {checked} packs, {failures} failures")
sys.exit(0 if failures == 0 else 1)
