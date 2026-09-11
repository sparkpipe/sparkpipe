#!/usr/bin/env python3
"""hy4 FP8 rung: emit the flat tensor manifest the C cell walks.

Reads a placed rank pack's safetensors header, classifies every FP8
payload + U8 E8M0 scale pair through the scale-row-offset contract
(tools/hy4_fp8_scale_contract.py kind_of/contract_of, mirror of
SparkHy4Fp8ScaleContractOf) and writes one line per plane:
  il kind_code payload_offset scale_offset rows_outer rows_inner cols
  groups rule scale_stride row_offset group_offset spine
rows_outer = stored leading dim (experts for 3D expert tensors, 1 for
2D), rows_inner = the per-expert row count, groups = cols / 32,
scale_stride = the contract row pitch (global groups, o_proj 512),
row_offset/group_offset = the rank base offsets of the contract byte
index (row_offset + row) * stride + group_offset + group. rule codes:
0 ALIGNED, 1 REPLICATED_ROWS, 2 REPLICATED_GROUPS. spine codes:
1 SPINE, 0 EXPERT. mtp layers take il = 1000 + layer so plane keys stay
unique. Offsets are ABSOLUTE file offsets (8 + header_len +
data_offset). Any pair the contract rejects is a hard error.
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from hy4_fp8_scale_contract import (
    EXPERT_KINDS,
    contract_of,
    kind_of,
)

RULE_CODES = {
    "ALIGNED": 0,
    "REPLICATED_ROWS": 1,
    "REPLICATED_GROUPS": 2,
}


def kind_name(name):
    for prefix in ("model.layers.", "model.mtp_layers."):
        if prefix in name:
            return name.split(prefix, 1)[1].split(".", 1)[1]
    return name[len("model."):] if name.startswith("model.") else name


def plane_index(name):
    if name.startswith("model.mtp_layers."):
        return 1000 + int(name.split("model.mtp_layers.", 1)[1].split(".", 1)[0])
    if name.startswith("model.layers."):
        return int(name.split("model.layers.", 1)[1].split(".", 1)[0])
    return -1


def main():
    if len(sys.argv) != 5:
        print("usage: fp8_rung_manifest.py PACK OUT RANK RANKS",
              file=sys.stderr)
        return 2
    pack, out = sys.argv[1], sys.argv[2]
    rank, ranks = int(sys.argv[3]), int(sys.argv[4])
    with open(pack, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data_start = 8 + n
    pairs = []
    failures = []
    for name, rec in header.items():
        if name == "__metadata__" or rec["dtype"] != "F8_E4M3":
            continue
        scale = header.get(name + "_scale")
        if scale is None or scale["dtype"] != "U8":
            failures.append((name, "no-scale"))
            continue
        shape = rec["shape"]
        if len(shape) not in (2, 3):
            failures.append((name, "rank%d" % len(shape)))
            continue
        kind, _, _ = kind_of(name)
        if kind is None:
            failures.append((name, "no-kind"))
            continue
        prow = 1
        for d in shape[:-1]:
            prow *= d
        pcol = shape[-1]
        got = contract_of(kind, rank, ranks, prow, pcol)
        if isinstance(got, str):
            failures.append((name, got))
            continue
        rule, scale_rows, scale_groups, row_off, group_off = got
        sshape = scale["shape"]
        srow = 1
        for d in sshape[:-1]:
            srow *= d
        if srow != scale_rows or sshape[-1] != scale_groups:
            failures.append((name, "scale %s want %dx%d" %
                             (sshape, scale_rows, scale_groups)))
            continue
        pairs.append((name, shape, rec, scale, rule, scale_groups,
                      row_off, group_off))
    if failures:
        for name, why in failures:
            print("CONTRACT FAIL %s: %s" % (name, why), file=sys.stderr)
        print("manifest %s: %d contract failures" % (out, len(failures)),
              file=sys.stderr)
        return 1
    names = sorted(set(kind_name(p[0]) for p in pairs))
    codes = {n: i for i, n in enumerate(names)}
    seen = {}
    lines = []
    for name, shape, rec, scale, rule, stride, row_off, group_off \
            in sorted(pairs):
        cols = shape[-1]
        if len(shape) == 3:
            outer, inner = shape[0], shape[1]
        else:
            outer, inner = 1, shape[0]
        il = plane_index(name)
        key = (il, codes[kind_name(name)])
        if key in seen:
            print("manifest %s: duplicate plane key %s for %s and %s" %
                  (out, key, seen[key], name), file=sys.stderr)
            return 1
        seen[key] = name
        lines.append("%d %d %d %d %d %d %d %d %d %d %d %d %d" % (
            il, codes[kind_name(name)],
            data_start + rec["data_offsets"][0],
            data_start + scale["data_offsets"][0],
            outer, inner, cols, cols // 32,
            RULE_CODES[rule], stride, row_off, group_off,
            0 if kind_of(name)[0] in EXPERT_KINDS else 1))
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    spine = sum(1 for line in lines if line.split()[-1] == "1")
    repl = sum(1 for line in lines if line.split()[8] != "0")
    print("manifest %s: %d planes (%d spine, %d expert), %d kinds" % (
        out, len(lines), spine, len(lines) - spine, len(names)))
    print("  rules: %d aligned, %d replicated" % (len(lines) - repl, repl))
    for n in names:
        print("  kind %d %s" % (codes[n], n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
