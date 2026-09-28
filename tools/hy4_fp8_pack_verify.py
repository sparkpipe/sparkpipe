#!/usr/bin/env python3
"""Independent verification of a hy4 FP8 TP16 rank pack.

Byte-compares sampled tensors of a built rank pack against the source
checkpoint (safetensors data_offsets are resolved relative to each shard's
own data section). Samples every slice kind: dim0 range, dim1 gather,
replicate, plus MTP and scale companions.

Usage:
  python3 tools/hy4_fp8_pack_verify.py --checkpoint DIR \
      --pack-dir DIR --rank 7 [--samples 6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

CHUNK = 1 << 20


def shard_stream(checkpoint: Path, shard: str, data_start: int):
    f = open(checkpoint / shard, "rb")

    def read_range(off, nbytes):
        f.seek(data_start + off)
        return f.read(nbytes)

    return read_range, f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--pack-dir", type=Path, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--rows-per-tensor", type=int, default=0)
    ap.add_argument("--skip-sha", action="store_true")
    args = ap.parse_args(argv)

    index = json.load(open(args.checkpoint / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    data_starts = {}
    headers = {}
    for shard in sorted(set(weight_map.values())):
        with open(args.checkpoint / shard, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            headers[shard] = json.loads(f.read(n))
            data_starts[shard] = 8 + n

    pack_path = (args.pack_dir /
                 f"model-fp8-tp16-rank-{args.rank:02d}.safetensors")
    pf = open(pack_path, "rb")
    (pn,) = struct.unpack("<Q", pf.read(8))
    phdr = json.loads(pf.read(pn))
    pbase = 8 + pn

    manifest = json.load(open(args.pack_dir /
                              f"manifest-rank-{args.rank:02d}.json"))
    if not args.skip_sha:
        digest = hashlib.sha256()
        pf.seek(0)
        for chunk in iter(lambda: pf.read(1 << 22), b""):
            digest.update(chunk)
        if manifest.get("file_sha256") != digest.hexdigest():
            print("SHA MISMATCH vs manifest")
            return 1
        print(f"sha OK ({digest.hexdigest()[:16]})")

    readers = {}

    def src_reader(shard):
        if shard not in readers:
            readers[shard] = shard_stream(args.checkpoint, shard,
                                          data_starts[shard])
        return readers[shard][0]

    if args.rows_per_tensor > 0:
        return verify_rows(manifest, phdr, pbase, pf, weight_map, headers,
                           src_reader, args.rank, args.rows_per_tensor)

    groups = {"range": [], "gather": [], "full": []}
    for t in manifest["tensors"]:
        sl = t["slice"]
        kind = "full" if sl == "replicate" else sl.get("kind")
        groups[kind].append(t)

    bad = 0
    picks = []
    for kind in ("range", "gather", "full"):
        picks += [(kind, t) for t in groups[kind][::max(
            1, len(groups[kind]) // args.samples)][:args.samples]]
    mtp = [t for t in manifest["tensors"] if "mtp_layers" in t["name"]]
    if mtp:
        picks.append(("mtp", mtp[len(mtp) // 2]))
    scales = [t for t in manifest["tensors"]
              if t["name"].endswith(("weight_scale", "_scale"))]
    picks.append(("scale", scales[len(scales) // 2]))

    for kind, t in picks:
        name = t["name"]
        ph = phdr[name]
        a, b = ph["data_offsets"]
        pf.seek(pbase + a)
        got = pf.read(b - a)
        shard = weight_map[name]
        meta = headers[shard][name]
        d0, d1 = meta["data_offsets"]
        sl = t["slice"]
        if sl == "replicate":
            want = src_reader(shard)(d0, d1 - d0)
        elif sl["kind"] == "range":
            esize = 1 if t["dtype"] in ("F8_E4M3", "U8") else \
                (2 if t["dtype"] == "BF16" else 4)
            row = esize
            for d in t["dims"][1:]:
                row *= d
            start = sl["start"] * row
            want = src_reader(shard)(d0 + start, b - a)
        else:
            dim = sl["dim"]
            chunk = t["dims"][dim] // 16
            esize = 1 if t["dtype"] in ("F8_E4M3", "U8") else \
                (2 if t["dtype"] == "BF16" else 4)
            lo, hi = sl["start"] * esize, (sl["start"] + chunk) * esize
            rows = 1
            for d in t["dims"][:-1]:
                rows *= d
            if dim == 0:
                want = src_reader(shard)(d0 + lo, b - a)
            else:
                row = esize * t["dims"][-1]
                full = src_reader(shard)(d0, rows * row)
                parts = []
                for r in range(rows):
                    parts.append(full[r * row + lo:r * row + hi])
                want = b"".join(parts)
        if got != want:
            print(f"{kind:6s} {name}: MISMATCH "
                  f"({sum(1 for x, y in zip(got, want) if x != y)} bytes)")
            bad += 1
        else:
            print(f"{kind:6s} {name}: OK ({len(got)} bytes)")
    print("VERIFY", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def element_size(dtype: str) -> int:
    return 1 if dtype in ("F8_E4M3", "U8") else (2 if dtype == "BF16" else 4)


def verify_rows(manifest, phdr, pbase, pf, weight_map, headers, src_reader,
                rank, rows_per_tensor) -> int:
    bad = 0
    checked = 0
    for t in manifest["tensors"]:
        name = t["name"]
        esize = element_size(t["dtype"])
        dims = t["dims"]
        columns = dims[-1]
        source_rows = 1
        for d in dims[:-1]:
            source_rows *= d
        sl = t["slice"]
        gather = sl != "replicate" and len(dims) > 1 and sl["dim"] == len(dims) - 1
        a, b = phdr[name]["data_offsets"]
        d0, _ = headers[weight_map[name]][name]["data_offsets"]
        read_source = src_reader(weight_map[name])
        if sl == "replicate":
            pack_columns, pack_rows, base, column_start = columns, source_rows, 0, 0
        elif sl["dim"] == 0:
            inner = esize
            for d in dims[1:]:
                inner *= d
            if len(dims) == 1:
                pack_columns, pack_rows = (b - a) // esize, 1
                base = sl["start"] * esize
            else:
                pack_columns, pack_rows = columns, (b - a) // (columns * esize)
                base = sl["start"] * inner
            column_start = 0
        elif gather:
            pack_columns = columns // 16
            pack_rows, base, column_start = source_rows, 0, sl["start"]
        else:
            print(f"{name}: unsupported slice {sl}")
            bad += 1
            continue
        if pack_rows * pack_columns * esize != b - a:
            print(f"{name}: pack extent {b - a} disagrees with {pack_rows}x{pack_columns}x{esize}")
            bad += 1
            continue
        picks = sorted({0, pack_rows - 1} | {(pack_rows * (i + 1)) // (rows_per_tensor + 1) for i in range(rows_per_tensor)})
        for row in picks:
            pf.seek(pbase + a + row * pack_columns * esize)
            got = pf.read(pack_columns * esize)
            if gather:
                offset = d0 + row * columns * esize + column_start * esize
            else:
                offset = d0 + base + row * pack_columns * esize
            want = read_source(offset, pack_columns * esize)
            checked += 1
            if got != want:
                print(f"{name} row {row}: MISMATCH")
                bad += 1
                break
    print(f"rank {rank}: {len(manifest['tensors'])} tensors, {checked} rows compared, {bad} mismatches")
    print("VERIFY", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
