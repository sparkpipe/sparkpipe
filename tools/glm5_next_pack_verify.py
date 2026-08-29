#!/usr/bin/env python3
"""Verify a glm5_next TP16 rank pack (.g5nsp) against the fixed packer's plan.

Checks (in order):
  1. header: magic, format, entry geometry, tp_degree/tp_rank, header
     file_bytes == actual file size == expected byte count (rank 0's receipt).
  2. directory: entry count == 1160; per-entry offset alignment and bounds;
     payload/scale byte counts consistent with (rows, cols, codec).
  3. plan diff: rebuild the Packer plan (NO payload production) from the
     checkpoint and compare every entry field in pack order — catches any
     packer drift (the 4758e68 class: fused section_slices (start, count)).
  4. spot round-trip (the kda lane's R6 pattern): regenerate the payload
     bytes for selected entries straight from the checkpoint through the
     packer's own produce closures and compare sha256 against the pack.

Default spot set: L17 kda_qkv_beta (the fused q|k|v|beta shard contract),
embedding rows shard, lm_head rows shard. --deep adds: L17 kda_decay_gate_
down (f|g), L0 dense up|gate, L3 shared up|gate, L3 q_b rows, L3 expert
up|gate slab. Every run also prints the directory sha256 so the 16 rank
inventories can be compared across nodes.

Run on the node holding the pack, with the checkpoint mounted:

  python3 tools/glm5_next_pack_verify.py \
      --pack ~/glm53_packs_fixed/glm5_next_stage.tp16.rank7.g5nsp \
      --source /mnt/model-warm/glm-5.3-flash --tp-rank 7 [--deep]

Exit 0 = all checks PASS; 1 = any failure (each failure printed).
"""
from __future__ import annotations

import argparse
import hashlib
import mmap
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm5_next_resident_stagepack import (  # noqa: E402
    ALIGNMENT, ENTRY_BYTES, FORMAT_VERSION, GLOBAL_LAYER, HEADER_BYTES, MAGIC,
    LAYERS, Packer, SCALE_F32, SCALE_NONE, SourceReader,
    K_DENSE_GATE_UP, K_EMBEDDING, K_EXPERT_UP_GATE, K_KDA_DECAY_GATE_DOWN,
    K_KDA_QKV_BETA, K_LM_HEAD, K_Q_B, K_SHARED_GATE_UP,
    PAYLOAD_BF16, PAYLOAD_F32, PAYLOAD_PACKED_WEIGHT,
)

EXPECTED_TENSOR_COUNT = 1160
EXPECTED_FILE_BYTES = 21706046976  # rank 0's completed receipt (fixed packer)

DT_SIZE = {PAYLOAD_BF16: 2, PAYLOAD_F32: 4, PAYLOAD_PACKED_WEIGHT: 1}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def parse_header(mm: mmap.mmap) -> dict:
    fields = struct.unpack_from("<20I", mm, 0)
    dir_off, file_bytes = struct.unpack_from("<QQ", mm, 80)
    keys = ["magic", "format_version", "header_bytes", "entry_bytes",
            "codec_abi_version", "flags", "entry_count", "stage_count",
            "stage_index", "first_layer", "layer_count", "total_layers",
            "hidden", "vocab", "experts", "codec1", "expert_codec",
            "codec2", "tp_degree", "tp_rank"]
    h = dict(zip(keys, fields))
    h["directory_offset"] = dir_off
    h["file_bytes"] = file_bytes
    return h


def parse_entries(mm: mmap.mmap, directory_offset: int, count: int) -> list:
    entries = []
    for i in range(count):
        off = directory_offset + i * ENTRY_BYTES
        e = struct.unpack_from("<8I4Q", mm, off)
        entries.append({
            "kind": e[0], "layer": e[1], "payload_type": e[2],
            "weight_codec": e[3], "scale_encoding": e[4],
            "group_count": e[5], "rows": e[6], "columns": e[7],
            "payload_offset": e[8], "payload_bytes": e[9],
            "scale_offset": e[10], "scale_bytes": e[11],
        })
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tp-rank", type=int, required=True)
    ap.add_argument("--tp-degree", type=int, default=16)
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args()

    path = Path(args.pack)
    size = path.stat().st_size
    print(f"pack: {path} ({size} bytes)")

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    h = parse_header(mm)
    if h["magic"] != MAGIC:
        fail(f"magic {h['magic']:#x} != {MAGIC:#x}")
    if h["format_version"] != FORMAT_VERSION:
        fail(f"format {h['format_version']}")
    if h["tp_degree"] != args.tp_degree or h["tp_rank"] != args.tp_rank:
        fail(f"header tp{h['tp_degree']} rank {h['tp_rank']} != "
             f"tp{args.tp_degree} rank {args.tp_rank}")
    if h["file_bytes"] != size:
        fail(f"header file_bytes {h['file_bytes']} != actual {size}")
    if size != EXPECTED_FILE_BYTES:
        fail(f"byte count {size} != rank-0 receipt {EXPECTED_FILE_BYTES}")
    print(f"PASS header: tp{h['tp_degree']} rank {h['tp_rank']}, "
          f"file_bytes {size} == rank-0 receipt")

    if h["entry_count"] != EXPECTED_TENSOR_COUNT:
        fail(f"entry count {h['entry_count']} != {EXPECTED_TENSOR_COUNT}")
    print(f"PASS count: {h['entry_count']} tensors")

    entries = parse_entries(mm, h["directory_offset"], h["entry_count"])
    end_max = 0
    for i, e in enumerate(entries):
        dsz = DT_SIZE.get(e["payload_type"])
        if dsz is None:
            fail(f"entry {i}: payload_type {e['payload_type']}")
        if e["payload_offset"] % ALIGNMENT:
            fail(f"entry {i}: payload offset {e['payload_offset']} unaligned")
        end = e["payload_offset"] + e["payload_bytes"]
        if e["scale_bytes"]:
            if e["scale_offset"] % ALIGNMENT:
                fail(f"entry {i}: scale offset unaligned")
            if e["scale_offset"] < end:
                fail(f"entry {i}: scale region overlaps payload")
            end = e["scale_offset"] + e["scale_bytes"]
        if end > size:
            fail(f"entry {i}: region end {end} > file {size}")
        end_max = max(end_max, end)
        if e["scale_encoding"] == SCALE_NONE and e["scale_bytes"]:
            fail(f"entry {i}: scale_bytes with SCALE_NONE")
        if e["scale_encoding"] == SCALE_F32 and e["scale_bytes"]:
            want = (e["group_count"] * e["rows"] * (e["columns"] // 128) * 4)
            if e["payload_type"] == PAYLOAD_PACKED_WEIGHT and e["scale_bytes"] != want:
                fail(f"entry {i}: scale_bytes {e['scale_bytes']} != {want}")
        if not e["payload_bytes"]:
            fail(f"entry {i} kind={e['kind']} layer={e['layer']:#x}: "
                 f"EMPTY payload (the 4758e68 fused-slice signature)")
        if e["payload_type"] in (PAYLOAD_BF16, PAYLOAD_F32, PAYLOAD_PACKED_WEIGHT):
            # kv_b entries carry group_count=MLA_HEADS; experts carry
            # group_count=EXPERTS — the payload is always group*rows*cols*dsz.
            want = e["group_count"] * e["rows"] * e["columns"] * dsz
            if e["payload_bytes"] != want:
                fail(f"entry {i} kind={e['kind']} layer={e['layer']:#x}: "
                     f"payload {e['payload_bytes']} != group*rows*cols {want}")
    if end_max != size:
        fail(f"last region end {end_max} != file size {size} (trailing bytes)")
    print("PASS layout: offsets aligned+bounded, payload sizes consistent, "
          "no empty payloads")

    dir_sha = hashlib.sha256(
        mm[h["directory_offset"]:h["directory_offset"] + h["entry_count"] * ENTRY_BYTES]
    ).hexdigest()
    print(f"directory sha256: {dir_sha}")

    # -- plan diff against the fixed packer (headers-only; no payload reads
    #    except three small f32 vectors the packer reads at plan time) -----
    source = SourceReader(Path(args.source))
    packer = Packer(source, args.tp_degree, args.tp_rank, 0, LAYERS,
                    False, True, True)
    packer.build()
    want = [{
        "kind": it.entry.kind, "layer": it.entry.layer,
        "payload_type": it.entry.payload_type,
        "weight_codec": it.entry.weight_codec,
        "scale_encoding": it.entry.scale_encoding,
        "group_count": it.entry.group_count, "rows": it.entry.rows,
        "columns": it.entry.columns, "payload_bytes": it.entry.payload_bytes,
        "scale_bytes": it.entry.scale_bytes,
    } for it in packer.plan]
    if len(want) != len(entries):
        fail(f"plan entry count {len(want)} != pack {len(entries)}")
    mismatches = 0
    for i, (w, e) in enumerate(zip(want, entries)):
        if w != e:
            mismatches += 1
            if mismatches <= 5:
                print(f"FAIL entry {i}: pack {e} != plan {w}")
    if mismatches:
        fail(f"plan diff: {mismatches}/{len(entries)} entries differ")
    print(f"PASS plan diff: all {len(entries)} entries match the fixed "
          f"packer's plan for rank {args.tp_rank}")

    # -- spot round-trip ---------------------------------------------------
    spot = [(K_KDA_QKV_BETA, 17), (K_KDA_DECAY_GATE_DOWN, 17),
            (K_DENSE_GATE_UP, 0), (K_EMBEDDING, GLOBAL_LAYER)]
    if args.deep:
        spot += [(K_LM_HEAD, GLOBAL_LAYER), (K_SHARED_GATE_UP, 3),
                 (K_Q_B, 3), (K_EXPERT_UP_GATE, 3)]
    by_key = {(it.entry.kind, it.entry.layer): it for it in packer.plan}
    for kind, layer in spot:
        item = by_key.get((kind, layer))
        if item is None:
            fail(f"spot: plan has no (kind={kind}, layer={layer:#x})")
        e = entries[[ (x["kind"], x["layer"]) for x in entries ].index((kind, layer))]
        produced = hashlib.sha256(b"".join(item.produce_payload())).hexdigest()
        onpack = hashlib.sha256(
            mm[e["payload_offset"]:e["payload_offset"] + e["payload_bytes"]]
        ).hexdigest()
        label = f"kind={kind} layer={layer:#x} ({e['payload_bytes']} B)"
        if produced != onpack:
            fail(f"spot round-trip {label}: pack {onpack[:16]} != ckpt {produced[:16]}")
        print(f"PASS spot round-trip {label}: sha {onpack[:16]}")

    source.close()
    mm.close()
    f.close()
    print(f"VERIFY-PASS rank {args.tp_rank}: {path.name} "
          f"{size} bytes, {h['entry_count']} tensors, dir_sha {dir_sha[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
