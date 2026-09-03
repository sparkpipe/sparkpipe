#!/usr/bin/env python3
"""hy4 lane: stage 1 (v2, empirical) of the FP8 pack header repair — derives
each tensor's ACTUAL data offset in the pack by content search and prints the
rebuilt safetensors prefix (8-byte length + JSON blob space-padded to the old
blob length) to stdout as binary; the caller applies it with dd conv=notrunc.

Why empirical: the fanout writer laid data out contiguously, but per-tensor
actual write sizes can exceed the declared sizes (source tensors whose
end-start exceeds esize*dims for replicate), so no closed-form layout matches
the file. Instead each tensor is located by searching the pack for a probe
taken from its expected source content, starting from the running position
(actual drift is bounded by a few tens of KB over the whole layout).

Tensor order = manifest order (the writer's plan order). Slice-to-source
offset math mirrors hy4_fp8_pack_verify.py.

Usage:
  python3 tools/hy4_fp8_header_repair2.py --pack PACK \
      --checkpoint DIR --manifest MANIFEST  > prefix.bin
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
from pathlib import Path

PROBE = 128


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--max-drift", type=int, default=1 << 20)
    args = ap.parse_args(argv)

    index = json.load(open(args.checkpoint / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    headers = {}
    data_starts = {}
    for shard in sorted(set(weight_map.values())):
        with open(args.checkpoint / shard, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            headers[shard] = json.loads(f.read(n))
            data_starts[shard] = 8 + n
    manifest = json.load(open(args.manifest))

    with open(args.pack, "rb") as f:
        (hl,) = struct.unpack("<Q", f.read(8))
        old_blob = f.read(hl)
    data_base = 8 + hl
    pack_fd = os.open(args.pack, os.O_RDONLY)
    data = mmap.mmap(pack_fd, 0, access=mmap.ACCESS_READ)

    esize_map = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "U8": 1}

    def src_probe(t, off_in_out, nbytes):
        name = t["name"]
        shard = weight_map[name]
        meta = headers[shard][name]
        d0 = meta["data_offsets"][0]
        sl = t["slice"]
        esize = esize_map[meta["dtype"]]
        if sl == "replicate":
            src_off = d0 + off_in_out
        elif sl["kind"] == "range":
            row = esize
            for d in t["dims"][1:]:
                row *= d
            src_off = d0 + sl["start"] * row + off_in_out
        else:
            dim = sl["dim"]
            esz = esize
            lo = sl["start"] * esz
            row = esz * t["dims"][-1]
            rows = 1
            for d in t["dims"][:-1]:
                rows *= d
            per = (t["dims"][dim] // 16) * esz
            row_idx = off_in_out // per
            within = off_in_out % per
            src_off = d0 + row_idx * row + lo + within
        with open(args.checkpoint / shard, "rb") as f:
            f.seek(data_starts[shard] + src_off)
            return f.read(nbytes)

    old_hdr = json.loads(old_blob)
    new_offsets = {}
    running = 0
    misses = 0
    walked = 0
    for t in manifest["tensors"]:
        name = t["name"]
        nbytes = old_hdr[name]["data_offsets"][1] - \
            old_hdr[name]["data_offsets"][0]
        head = src_probe(t, 0, min(64, nbytes))
        expected = data_base + running
        actual = None
        if bytes(data[expected:expected + len(head)]) == head:
            actual = running
        elif nbytes > (1 << 20):
            lo = max(0, data_base + running - args.max_drift)
            pos = data.find(head, lo,
                            data_base + running + args.max_drift + len(head))
            if pos >= 0:
                mid_in = nbytes // 2
                mid = src_probe(t, mid_in, min(64, nbytes - mid_in))
                if bytes(data[pos + mid_in:pos + mid_in + len(mid)]) == mid:
                    actual = pos - data_base
        if actual is None:
            lo = data_base + running
            first = data.find(head, lo, lo + args.max_drift + len(head))
            if first >= 0 and \
                    data.find(head, first + 1,
                              lo + args.max_drift + len(head)) < 0:
                actual = first - data_base
        if actual is None:
            print(f"NO-MATCH {name}", file=sys.stderr)
            misses += 1
            continue
        new_offsets[name] = [actual, actual + nbytes]
        running = actual + nbytes
        walked += 1
        if walked % 128 == 0:
            try:
                data.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError):
                pass
    if misses:
        raise SystemExit(f"{misses} tensors could not be located; "
                         f"prefix NOT emitted")

    new_hdr = {"__metadata__": old_hdr.get("__metadata__", {})}
    for t in manifest["tensors"]:
        name = t["name"]
        ent = dict(old_hdr[name])
        ent["data_offsets"] = new_offsets[name]
        new_hdr[name] = ent
    new_blob = json.dumps(new_hdr, separators=(",", ":")).encode()
    if len(new_blob) > hl:
        raise SystemExit(f"new header {len(new_blob)} > old {hl}")
    new_blob += b" " * (hl - len(new_blob))
    sys.stdout.buffer.write(struct.pack("<Q", hl))
    sys.stdout.buffer.write(new_blob)
    sys.stdout.buffer.flush()
    print(f"located {len(new_offsets)} tensors empirically, "
          f"last data end {running}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
