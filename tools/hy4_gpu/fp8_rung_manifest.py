#!/usr/bin/env python3
"""hy4 FP8 rung: emit the flat tensor manifest the C cell walks.

Reads a placed rank pack's safetensors header and writes one line per
FP8 payload + E8M0 scale pair (expert gate_up/down and the per-layer
attention projections alike):
  il kind_code payload_offset scale_offset rows_outer rows_inner cols groups
rows_outer = stored leading dim (experts for 3D expert tensors, 1 for 2D),
rows_inner = the per-expert row count so the cell's sampling is uniform,
groups = cols / 32 (E8M0 group-32 scales). Offsets are ABSOLUTE file
offsets (8 + header_len + data_offset). kind_code indexes the sorted
distinct projection names.
"""
import json
import struct
import sys


def kind_name(name):
    for prefix in ("model.layers.", "model.mtp_layers."):
        if prefix in name:
            return name.split(prefix, 1)[1].split(".", 1)[1]
    return name[len("model."):] if name.startswith("model.") else name


def main():
    pack, out = sys.argv[1], sys.argv[2]
    with open(pack, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data_start = 8 + n
    pairs = []
    unmatched = []
    for name, rec in header.items():
        if name == "__metadata__" or rec["dtype"] != "F8_E4M3":
            continue
        scale = header.get(name + "_scale")
        if scale is None or scale["dtype"] != "U8":
            unmatched.append((name, "no-scale"))
            continue
        shape = rec["shape"]
        sshape = scale["shape"]
        if len(shape) not in (2, 3):
            unmatched.append((name, "rank%d" % len(shape)))
            continue
        cols = shape[-1]
        if sshape != shape[:-1] + [cols // 32]:
            unmatched.append((name, "shape %s vs %s" % (shape, sshape)))
            continue
        pairs.append((name, shape, rec, scale))
    names = sorted(set(kind_name(p) for p, _, _, _ in pairs))
    codes = {n: i for i, n in enumerate(names)}
    lines = []
    for name, shape, rec, scale in sorted(pairs):
        cols = shape[-1]
        if len(shape) == 3:
            outer, inner = shape[0], shape[1]
        else:
            outer, inner = 1, shape[0]
        il = -1
        if ".layers." in name:
            il = int(name.split(".layers.")[1].split(".")[0])
        lines.append("%d %d %d %d %d %d %d %d" % (
            il, codes[kind_name(name)],
            data_start + rec["data_offsets"][0],
            data_start + scale["data_offsets"][0],
            outer, inner, cols, cols // 32))
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("manifest %s: %d FP8 pairs, %d kinds, %d unmatched" % (
        out, len(lines), len(names), len(unmatched)))
    for name, why in unmatched:
        print("  UNMATCHED %s: %s" % (name, why))
    for n in names:
        print("  kind %d %s" % (codes[n], n))


if __name__ == "__main__":
    main()
