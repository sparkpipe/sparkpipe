#!/usr/bin/env python3
"""Patch a placed qwen38max nvfp4 TP16 pack to the explicit codec contract.

The placed packs already carry the exact nvfp4 payload bytes and per-16 e4m3
scale planes the fixed builder writes; the byte-level delta is metadata
(entry weight_format 8, scale_group_size 16, scale_bytes + resident*8, the
shifted offsets that follow) plus the per-expert
[input_scale][weight_scale_2] f32 tails - 8 bytes inside each expert's scale
segment, gathered from the warm checkpoint by tensor name (141 KB/rank).
Proof obligation: patching rank 0's replaced pack is byte-identical to the
true warm rebuild of the same rank.
"""
import argparse
import hashlib
import json
import pathlib
import struct

HEADER_BYTES = 128
ENTRY_BYTES = 56
ENTRY_STRUCT = struct.Struct("<6I4Q")
KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN = 6, 7, 8
WEIGHT_FP8_F32B128, WEIGHT_NVFP4_PACKED, WEIGHT_BF16 = 4, 8, 0
ALIGN = 256
EXPERT_INTERMEDIATE, HIDDEN, EXPERT_COUNT = 2048, 8192, 512
KIND_TO_PROJ = {KIND_MOE_W1: "gate_proj", KIND_MOE_W3: "up_proj",
                KIND_MOE_DOWN: "down_proj"}


def expert_rows_of(kind):
    return HIDDEN if kind == KIND_MOE_DOWN else EXPERT_INTERMEDIATE


def is_expert(kind):
    return kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN)


def load_tails(checkpoint, names):
    index = json.load(open(checkpoint / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    wanted = {name: None for name in names}
    per_shard = {}
    for name in wanted:
        per_shard.setdefault(weight_map[name], []).append(name)
    for shard, shard_names in per_shard.items():
        with open(checkpoint / shard, "rb") as handle:
            handle.seek(0)
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
            base = 8 + length
            spans = sorted((header[name]["data_offsets"][0], name)
                           for name in shard_names)
            handle.seek(base + spans[0][0])
            cursor = base + spans[0][0]
            for offset, name in spans:
                if cursor != base + offset:
                    handle.seek(base + offset)
                wanted[name] = handle.read(4)
                cursor = base + offset + 4
    missing = [n for n, v in wanted.items() if not v or len(v) != 4]
    if missing:
        raise ValueError(f"tail reads failed: {missing[:3]}")
    return wanted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-pack", required=True)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    old = open(args.old_pack, "rb")
    header = old.read(HEADER_BYTES)
    fields = struct.unpack("<28I2Q", header)
    magic, version, header_bytes, entry_bytes, tensor_count = fields[:5]
    if magic != 0x50533851 or version != 2 or header_bytes != HEADER_BYTES:
        raise ValueError("not a v2 qwen38 stagepack")
    if fields[26] != args.tp_degree or fields[27] != args.tp_rank:
        raise ValueError("header tp fields disagree with the invocation")
    entries = [ENTRY_STRUCT.unpack(old.read(ENTRY_BYTES))
               for _ in range(tensor_count)]
    directory_offset = fields[28]
    old_file_bytes = fields[29]
    payload_base = -(-((HEADER_BYTES + tensor_count * ENTRY_BYTES)) // ALIGN) * ALIGN

    new_entries = []
    tail_names = []
    for (kind, layer, fmt, rows, cols, group, p_off, p_bytes, s_off, s_bytes) in entries:
        if fmt == WEIGHT_FP8_F32B128 and is_expert(kind):
            expert_rows = expert_rows_of(kind)
            if rows % expert_rows != 0 or cols % 16 != 0:
                raise ValueError(f"kind {kind} slab {rows}x{cols} not expert-shaped")
            resident = rows // expert_rows
            first = args.tp_rank * resident
            proj = KIND_TO_PROJ[kind]
            for e in range(first, first + resident):
                base_name = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                tail_names.append(base_name + ".input_scale")
                tail_names.append(base_name + ".weight_scale_2")
            new_entries.append([kind, layer, WEIGHT_NVFP4_PACKED, rows, cols, 16,
                                p_bytes, s_bytes + resident * 8])
        else:
            new_entries.append([kind, layer, fmt, rows, cols, group,
                                p_bytes, s_bytes])

    tails = load_tails(args.checkpoint, tail_names) if tail_names else {}

    # layout walk (mirrors the builder): every payload offset aligned, scale
    # immediately after its payload, cursor carries across entries.
    offsets = []
    cursor = payload_base
    for p_bytes, s_bytes in ((e[6], e[7]) for e in new_entries):
        p_off = -(-cursor // ALIGN) * ALIGN
        s_off = p_off + p_bytes if s_bytes else 0
        offsets.append((p_off, s_off))
        cursor = p_off + p_bytes + s_bytes
    file_bytes = cursor

    digest = hashlib.sha256()
    with open(args.out, "wb") as out:
        new_header = struct.pack("<28I2Q", *fields[:28], directory_offset, file_bytes)
        out.write(new_header)
        digest.update(new_header)
        for entry, (p_off, s_off) in zip(new_entries, offsets):
            blob = ENTRY_STRUCT.pack(entry[0], entry[1], entry[2], entry[3],
                                     entry[4], entry[5], p_off, entry[6],
                                     s_off, entry[7])
            out.write(blob)
            digest.update(blob)
        pad = payload_base - HEADER_BYTES - tensor_count * ENTRY_BYTES
        out.write(b"\0" * pad)
        digest.update(b"\0" * pad)

        for old_entry, new_entry, (p_off, s_off) in zip(entries, new_entries, offsets):
            (kind, layer, fmt, rows, cols, group, o_p_off, p_bytes, o_s_off, s_bytes) = old_entry
            copy_span(old, out, digest, o_p_off, p_bytes)
            if fmt == WEIGHT_FP8_F32B128 and is_expert(kind):
                expert_rows = expert_rows_of(kind)
                resident = rows // expert_rows
                plane = (rows // resident) * (cols // 16)
                first = args.tp_rank * resident
                proj = KIND_TO_PROJ[kind]
                for e in range(first, first + resident):
                    copy_span(old, out, digest, o_s_off + (e - first) * plane, plane)
                    base_name = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                    for suffix in (".input_scale", ".weight_scale_2"):
                        tail = tails[base_name + suffix]
                        out.write(tail)
                        digest.update(tail)
            elif s_bytes:
                copy_span(old, out, digest, o_s_off, s_bytes)

    actual = __import__("os").path.getsize(args.out)
    if actual != file_bytes:
        raise ValueError(f"size {actual} != walked {file_bytes}")
    print(json.dumps({"out": args.out, "bytes": actual,
                      "sha256": digest.hexdigest()}, sort_keys=True))


def copy_span(src, dst, digest, offset, length):
    src.seek(offset)
    remaining = length
    while remaining > 0:
        chunk = src.read(min(remaining, 1 << 22))
        if not chunk:
            raise ValueError(f"short read at {offset}+{length}")
        remaining -= len(chunk)
        dst.write(chunk)
        digest.update(chunk)


if __name__ == "__main__":
    main()
