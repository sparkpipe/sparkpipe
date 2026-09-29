#!/usr/bin/env python3
"""Print the SPARK_WEIGHTD_SPINE_BUDGET_BYTES a lazy attach of one rank pack needs.

The lazy attach fails closed unless the budget covers the compact spine
allocation the runtime derives from the pack's `.experts` range manifest
(runtime/spark_weightd_manifest.c build_spine) plus the 255-byte alignment
slack of runtime/spark_weightd_lazy_pack.c. This reproduces that derivation
from the same two files so launchers never pass a smaller smoke-manifest
figure.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

MAGIC = 0x58504557
VERSION = 2
HEADER = struct.Struct("<IIII")
RECORD = struct.Struct("<IIIIQQ16s")
ALIGNMENT_SLACK = 255


def spine_allocation(manifest: bytes, pack_bytes: int) -> tuple[int, int]:
    if len(manifest) < HEADER.size:
        raise ValueError("manifest shorter than its header")
    magic, version, count, reserved = HEADER.unpack_from(manifest, 0)
    if magic != MAGIC or version != VERSION or reserved != 0 or count == 0:
        raise ValueError("not a v2 expert range manifest")
    if len(manifest) != HEADER.size + count * RECORD.size:
        raise ValueError("manifest length does not match its range count")
    ranges = []
    for index in range(count):
        _, _, _, pad, offset, size, _ = RECORD.unpack_from(
            manifest, HEADER.size + index * RECORD.size)
        if pad != 0 or size == 0 or offset > pack_bytes or size > pack_bytes - offset:
            raise ValueError(f"range {index} is invalid for a {pack_bytes}-byte pack")
        ranges.append((offset, size))
    ranges.sort()
    for (left_offset, left_size), (right_offset, _) in zip(ranges, ranges[1:]):
        if right_offset < left_offset + left_size:
            raise ValueError("ranges overlap")
    cursor = 0
    allocation = 0
    spine = 0
    for offset, size in ranges + [(pack_bytes, 0)]:
        if offset > cursor:
            padding = (cursor - allocation) & 255
            compact = allocation + padding
            allocation = compact + (offset - cursor)
            spine += offset - cursor
        cursor = offset + size
    return spine, allocation


def budget_for(pack_path: str) -> dict:
    pack_bytes = os.stat(pack_path).st_size
    with open(pack_path + ".experts", "rb") as handle:
        spine, allocation = spine_allocation(handle.read(), pack_bytes)
    return {
        "pack": pack_path,
        "pack_bytes": pack_bytes,
        "spine_bytes": spine,
        "spine_allocation_bytes": allocation,
        "spine_budget_bytes": allocation + ALIGNMENT_SLACK,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", nargs="+")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)
    for pack in arguments.pack:
        record = budget_for(pack)
        if arguments.verbose:
            print(" ".join(f"{key}={value}" for key, value in record.items()))
        else:
            print(record["spine_budget_bytes"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
