#!/usr/bin/env python3
"""Write a control copy of a named stage pack with tensors deliberately corrupted.

The copy is an APFS clone (cp -c), so only the corrupted tensors' blocks take
new space. --tensor is a glob over the pack's tensor names, so
layers.31.ffn_down corrupts one tensor and layers.*.ffn_down the same tensor in
every layer, the reach of a kernel bug that every layer runs.
--corruption swap-rows swaps each tensor's first two rows, the size of a bug
that misplaces one output; rotate-rows moves every row down by one, the size
of a bug that offsets a whole tensor. The receipt records the tensors, the
corruption, and the sha256 of each tensor's bytes before and after. The copy
is a qualification control and is never served.

  pack_tensor_control.py --pack PACK --tensor GLOB --corruption swap-rows|rotate-rows --output CONTROL_PACK --receipt RECEIPT_JSON
"""

import argparse
import fnmatch
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

NAMED_PACK_MAGIC = 0x504E5053
NAMED_PACK_ALIGN = 128


def entry_locations(path, pattern):
    with path.open("rb") as handle:
        magic, version, manifest_bytes = struct.unpack("<IIQ", handle.read(16))
        if magic != NAMED_PACK_MAGIC:
            raise SystemExit(f"{path} is not a named stage pack")
        manifest = json.loads(handle.read(manifest_bytes))
    names = sorted(name for name in manifest["tensors"] if fnmatch.fnmatchcase(name, pattern))
    if not names:
        raise SystemExit(f"{path} has no tensor matching {pattern}")
    base = (16 + manifest_bytes + NAMED_PACK_ALIGN - 1) // NAMED_PACK_ALIGN * NAMED_PACK_ALIGN
    return [(name, base + manifest["tensors"][name]["offset"], manifest["tensors"][name]) for name in names]


def corrupt(handle, name, offset, entry, corruption):
    rows = entry["shape"][0]
    if rows < 2 or entry["bytes"] % rows != 0:
        raise SystemExit(f"{name} has no two whole rows to move")
    row_bytes = entry["bytes"] // rows
    handle.seek(offset)
    original = handle.read(entry["bytes"])
    if original[:row_bytes] == original[row_bytes:2 * row_bytes]:
        raise SystemExit(f"rows 0 and 1 of {name} are identical; moving them changes nothing")
    if corruption == "swap-rows":
        corrupted = original[row_bytes:2 * row_bytes] + original[:row_bytes] + original[2 * row_bytes:]
    else:
        corrupted = original[-row_bytes:] + original[:-row_bytes]
    handle.seek(offset)
    handle.write(corrupted)
    return {"tensor": name, "rows": rows, "row_bytes": row_bytes, "sha256_before": hashlib.sha256(original).hexdigest(),
            "sha256_after": hashlib.sha256(corrupted).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--corruption", choices=("swap-rows", "rotate-rows"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    locations = entry_locations(arguments.pack, arguments.tensor)
    subprocess.run(["cp", "-c", str(arguments.pack), str(arguments.output)], check=True)
    with arguments.output.open("r+b") as handle:
        tensors = [corrupt(handle, name, offset, entry, arguments.corruption) for name, offset, entry in locations]
    receipt = {"pack": str(arguments.pack), "control_pack": str(arguments.output), "pattern": arguments.tensor,
               "corruption": {"swap-rows": "rows 0 and 1 swapped", "rotate-rows": "every row moved down by one"}[arguments.corruption],
               "tensors": tensors}
    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"pattern": arguments.tensor, "tensors": len(tensors), "corruption": receipt["corruption"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
