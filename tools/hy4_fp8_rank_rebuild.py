#!/usr/bin/env python3
"""hy4 lane: segmented rebuild for FP8 TP16 rank packs whose 1-D dim0-split
tensors (learnable_sink_param) were never written by the fanout packer on
ranks 1-15: the packer's 1-D row_bytes fallback made the whole tensor a
single row owned by rank 0, so rank 0 over-wrote 256B into its 16B slot
(harmless once headers are repaired) while ranks 1-15 have 79 missing
16-byte holes and every following byte sits 16 bytes per hole too early.

Rebuild = locate every non-sink tensor's actual offset by content (same
empirical walk as hy4_fp8_header_repair2.py), then stream a new pack:
for each tensor in manifest order, emit the source slice bytes for a sink,
else copy the pack's actual bytes. The result has exactly the declared
layout, so the emitted header declares sequential offsets. Sidecar + the
manifest's file_sha256 are rewritten by the caller.

Usage:
  python3 tools/hy4_fp8_rank_rebuild.py --pack PACK --out OUT \
      --checkpoint DIR --manifest MANIFEST
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
from pathlib import Path

CHUNK = 1 << 20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
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
    old_hdr = json.loads(old_blob)
    data_base = 8 + hl
    pack_fd = os.open(args.pack, os.O_RDONLY)
    data = mmap.mmap(pack_fd, 0, access=mmap.ACCESS_READ)

    esize_map = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "U8": 1}

    shard_files = {}

    def src_bytes(t, off_in_out, nbytes):
        name = t["name"]
        shard = weight_map[name]
        if shard not in shard_files:
            shard_files[shard] = open(args.checkpoint / shard, "rb")
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
            lo = sl["start"] * esize
            row = esize * t["dims"][-1]
            rows = 1
            for d in t["dims"][:-1]:
                rows *= d
            per = (t["dims"][sl["dim"]] // 16) * esize
            row_idx = off_in_out // per
            within = off_in_out % per
            src_off = d0 + row_idx * row + lo + within
        f = shard_files[shard]
        f.seek(data_starts[shard] + src_off)
        return f.read(nbytes)

    def locate(t, nbytes):
        head = src_bytes(t, 0, min(64, nbytes))
        expected = data_base + running[0]
        if bytes(data[expected:expected + len(head)]) == head:
            running[0] += nbytes
            return expected - data_base
        if nbytes > (1 << 20):
            lo = max(0, data_base + running[0] - args.max_drift)
            pos = data.find(head, lo,
                            data_base + running[0] + args.max_drift +
                            len(head))
            if pos >= 0:
                mid_in = nbytes // 2
                mid = src_bytes(t, mid_in, min(64, nbytes - mid_in))
                if bytes(data[pos + mid_in:pos + mid_in + len(mid)]) == mid:
                    running[0] = pos - data_base + nbytes
                    return pos - data_base
        lo = data_base + running[0]
        first = data.find(head, lo, lo + args.max_drift + len(head))
        if first >= 0 and \
                data.find(head, first + 1,
                          lo + args.max_drift + len(head)) < 0:
            running[0] = first - data_base + nbytes
            return first - data_base
        return None

    running = [0]
    plan = []
    misses = 0
    for t in manifest["tensors"]:
        name = t["name"]
        nbytes = old_hdr[name]["data_offsets"][1] - \
            old_hdr[name]["data_offsets"][0]
        sl = t["slice"]
        is_sink = sl != "replicate" and sl["kind"] == "range" and \
            len(t["dims"]) == 1
        if is_sink:
            plan.append((name, None, nbytes))
            continue
        off = locate(t, nbytes)
        if off is None:
            print(f"NO-MATCH {name}", file=sys.stderr)
            misses += 1
            continue
        plan.append((name, off, nbytes))
    if misses:
        raise SystemExit(f"{misses} tensors could not be located; "
                         f"rebuild NOT started")
    try:
        data.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError):
        pass

    out_fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(out_fd, struct.pack("<Q", hl))
    os.ftruncate(out_fd, 8 + hl)
    new_hdr = {"__metadata__": old_hdr.get("__metadata__", {})}
    cursor = 0
    for name, off, nbytes in plan:
        new_hdr[name] = dict(old_hdr[name])
        new_hdr[name]["data_offsets"] = [cursor, cursor + nbytes]
        cursor += nbytes
    new_blob = json.dumps(new_hdr, separators=(",", ":")).encode()
    if len(new_blob) > hl:
        raise SystemExit(f"new header {len(new_blob)} > old {hl}")
    os.pwrite(out_fd, new_blob + b" " * (hl - len(new_blob)), 8)

    abspos = 8 + hl
    copied = 0
    last_drop = 0
    for name, off, nbytes in plan:
        if off is None:
            t = next(x for x in manifest["tensors"]
                     if x["name"] == name)
            done = 0
            while done < nbytes:
                step = min(CHUNK, nbytes - done)
                buf = src_bytes(t, done, step)
                if len(buf) != step:
                    raise SystemExit(f"short src read {name}")
                os.pwrite(out_fd, buf, abspos + done)
                done += step
        else:
            done = 0
            while done < nbytes:
                step = min(CHUNK, nbytes - done)
                buf = data[data_base + off + done:
                           data_base + off + done + step]
                if len(buf) != step:
                    raise SystemExit(f"short pack read {name}")
                os.pwrite(out_fd, buf, abspos + done)
                done += step
        abspos += nbytes
        copied += nbytes
        if copied - last_drop >= (1 << 27):
            last_drop = copied
            try:
                data.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError):
                pass
    os.fsync(out_fd)
    os.close(out_fd)
    print(f"rebuilt {len(plan)} tensors, {copied} data bytes "
          f"(pack data ends at {cursor})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
