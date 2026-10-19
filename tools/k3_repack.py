#!/usr/bin/env python3
"""K3 rank pack format 3: census, convert and verify.

check <v2.pack>  reads a format 2 cell-split rank pack with O_DIRECT and
                 reports, per expert tensor, the E8M0 scale codes of its
                 non-zero 32-element blocks and of its all-zero blocks, and
                 whether one 4-bit window per tensor holds every code."""
import json
import mmap
import os
import struct
import sys

import numpy as np

MAGIC = 0x4B33504B
SOURCE_VERSION = 2
ALIGN = 128
CELL_PAYLOAD_ROWS = 16
SOURCE_CELL_ROWS = 17
ROW_BYTES = 16
TILE_K = 32
SCALE_SPAN = 16
DIRECT = 1 << 20
CHUNK = 64 << 20


class RepackFailure(RuntimeError):
    pass


class DirectReader:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        self.size = os.fstat(self.fd).st_size
        self.buffer = mmap.mmap(-1, CHUNK + 2 * DIRECT)

    def read(self, offset, length):
        start = offset - offset % DIRECT
        end = offset + length
        span = end - start
        span += (-span) % DIRECT
        if span > len(self.buffer):
            raise RepackFailure(f"read of {length} bytes at {offset} exceeds the {CHUNK} byte window")
        view = memoryview(self.buffer)[:span]
        got = os.preadv(self.fd, [view], start)
        if got < min(span, self.size - start):
            raise RepackFailure(f"short read at {start}: {got} of {span} bytes")
        return bytes(view[offset - start:offset - start + length])

    def close(self):
        self.buffer.close()
        os.close(self.fd)


def load_manifest(reader):
    magic, version, length = struct.unpack("<IIQ", reader.read(0, 16))
    if magic != MAGIC:
        raise RepackFailure("not a K3 pack")
    if version != SOURCE_VERSION:
        raise RepackFailure(f"K3 pack format {version}; check reads format {SOURCE_VERSION}")
    manifest = json.loads(reader.read(16, length))
    base = 16 + length
    base += (-base) % ALIGN
    return manifest, base


def expert_tensors(manifest):
    out = []
    for name, entry in manifest["tensors"].items():
        if entry.get("kind") != "mxfp4_ws_interleaved_v1":
            continue
        geom = entry.get("interleave") or {}
        if geom.get("tile_k") != TILE_K or geom.get("row_bytes") != ROW_BYTES or geom.get("cell_rows") != SOURCE_CELL_ROWS:
            raise RepackFailure(f"{name}: interleave {geom.get('tile_k')}/{geom.get('row_bytes')}/{geom.get('cell_rows')} is not {TILE_K}/{ROW_BYTES}/{SOURCE_CELL_ROWS}")
        if geom["k_tiles"] % 2 != 0:
            raise RepackFailure(f"{name}: {geom['k_tiles']} k-tiles do not pair")
        out.append((name, entry, geom))
    if not out:
        raise RepackFailure("the pack has no interleaved expert tensors")
    return sorted(out, key=lambda item: item[1]["offset"])


def census_tensor(reader, base, entry, geom):
    experts = entry["shape"][0]
    expert_bytes = geom["expert_bytes"]
    if expert_bytes * experts != entry["bytes"]:
        raise RepackFailure(f"expert bytes {expert_bytes} x {experts} != tensor bytes {entry['bytes']}")
    per_chunk = max(1, CHUNK // expert_bytes)
    live = np.zeros(256, dtype=np.int64)
    zero = np.zeros(256, dtype=np.int64)
    for first in range(0, experts, per_chunk):
        count = min(per_chunk, experts - first)
        raw = reader.read(base + entry["offset"] + first * expert_bytes, count * expert_bytes)
        grid = np.frombuffer(raw, dtype=np.uint8).reshape(count, geom["k_tiles"], geom["cells"], SOURCE_CELL_ROWS, ROW_BYTES)
        scales = grid[:, :, :, CELL_PAYLOAD_ROWS, :]
        empty = np.all((grid[:, :, :, :CELL_PAYLOAD_ROWS, :] & 0x77) == 0, axis=-1)
        live += np.bincount(scales[~empty], minlength=256)
        zero += np.bincount(scales[empty], minlength=256)
    return live, zero


def window(live):
    codes = np.flatnonzero(live)
    if codes.size == 0:
        return None
    return int(codes[0]), int(codes[-1])


def check(path):
    reader = DirectReader(path)
    try:
        manifest, base = load_manifest(reader)
        config = manifest["config"]
        if config.get("expert_split") != "cells":
            raise RepackFailure(f"expert_split {config.get('expert_split')!r} is not cells")
        totals = {}
        ok = True
        for name, entry, geom in expert_tensors(manifest):
            live, zero = census_tensor(reader, base, entry, geom)
            span = window(live)
            fits = span is not None and span[0] >= 1 and span[1] <= 254 and span[1] - span[0] < SCALE_SPAN
            ok &= fits
            kind = "w1" if name.endswith("w1_weight") else "w2"
            totals.setdefault(kind, np.zeros(256, dtype=np.int64))
            totals[kind] += live
            print(json.dumps({"tensor": name, "low": span[0] if span else None, "high": span[1] if span else None,
                              "distinct": int(np.count_nonzero(live)), "zero_blocks": int(zero.sum()),
                              "zero_block_codes": [int(c) for c in np.flatnonzero(zero)], "fits": bool(fits)}), flush=True)
        summary = {kind: {"low": window(h)[0], "high": window(h)[1], "distinct": int(np.count_nonzero(h)),
                          "top": int(np.argmax(h)), "top_share": round(float(h.max() / h.sum()), 4)} for kind, h in totals.items()}
        print(json.dumps({"verdict": "convertible" if ok else "REFUSED", "rank": config.get("tp_rank"),
                          "degree": config.get("tp_degree"), "summary": summary}), flush=True)
        return 0 if ok else 3
    finally:
        reader.close()


def main(argv):
    if len(argv) == 3 and argv[1] == "check":
        return check(argv[2])
    print("usage: k3_repack.py check <v2.pack>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except RepackFailure as failure:
        print(f"k3_repack: {failure}", file=sys.stderr)
        sys.exit(1)
