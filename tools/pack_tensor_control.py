#!/usr/bin/env python3
"""Write a control copy of a named stage pack with one tensor deliberately corrupted.

The copy is an APFS clone (cp -c), so only the corrupted tensor's blocks take
new space. --corruption swap-rows swaps the tensor's first two rows, the size
of a packer bug that misplaces one row; rotate-rows moves every row down by
one, the size of a packer bug that offsets a whole tensor. The receipt
records the tensor, the corruption, and the sha256 of its bytes before and
after. The copy is a qualification control and is never served.

  pack_tensor_control.py --pack PACK --tensor NAME --corruption swap-rows|rotate-rows --output CONTROL_PACK --receipt RECEIPT_JSON
"""

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

NAMED_PACK_MAGIC = 0x504E5053
NAMED_PACK_ALIGN = 128


def entry_location(path, tensor):
    with path.open("rb") as handle:
        magic, version, manifest_bytes = struct.unpack("<IIQ", handle.read(16))
        if magic != NAMED_PACK_MAGIC:
            raise SystemExit(f"{path} is not a named stage pack")
        manifest = json.loads(handle.read(manifest_bytes))
    if tensor not in manifest["tensors"]:
        raise SystemExit(f"{path} has no tensor {tensor}")
    entry = manifest["tensors"][tensor]
    base = (16 + manifest_bytes + NAMED_PACK_ALIGN - 1) // NAMED_PACK_ALIGN * NAMED_PACK_ALIGN
    return base + entry["offset"], entry


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--corruption", choices=("swap-rows", "rotate-rows"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    offset, entry = entry_location(arguments.pack, arguments.tensor)
    rows = entry["shape"][0]
    if rows < 2 or entry["bytes"] % rows != 0:
        raise SystemExit(f"{arguments.tensor} has no two whole rows to swap")
    row_bytes = entry["bytes"] // rows
    subprocess.run(["cp", "-c", str(arguments.pack), str(arguments.output)], check=True)
    with arguments.output.open("r+b") as handle:
        handle.seek(offset)
        original = handle.read(entry["bytes"])
        if arguments.corruption == "swap-rows":
            corrupted = original[row_bytes:2 * row_bytes] + original[:row_bytes] + original[2 * row_bytes:]
        else:
            corrupted = original[-row_bytes:] + original[:-row_bytes]
        handle.seek(offset)
        handle.write(corrupted)
    if original[:row_bytes] == original[row_bytes:2 * row_bytes]:
        raise SystemExit(f"rows 0 and 1 of {arguments.tensor} are identical; swapping them changes nothing")
    receipt = {"pack": str(arguments.pack), "control_pack": str(arguments.output), "tensor": arguments.tensor,
               "corruption": {"swap-rows": "rows 0 and 1 swapped", "rotate-rows": "every row moved down by one"}[arguments.corruption],
               "row_bytes": row_bytes, "rows": rows,
               "tensor_sha256_before": hashlib.sha256(original).hexdigest(),
               "tensor_sha256_after": hashlib.sha256(corrupted).hexdigest()}
    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
