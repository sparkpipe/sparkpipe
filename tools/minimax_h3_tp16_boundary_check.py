#!/usr/bin/env python3
"""Boundary checks on the written minimax H3 TP16 rank packs (lane minimax).

Extents from the pack directories: 4-head ranks q/k/v rows 512, 3-head ranks
384; to_out columns mirror the same split; encoder k/v carry 128 rows at kv
head r//2; ffn 1792/896 rows-cols. Payload proof: the packed slice bytes must
hash equal to a fresh slice computed from the warm copy through the same plan
math, and the rank07+rank08 packed attention slices must concatenate to the
warm tensor's [3584,4480) middle.
"""

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import minimax_h3_stagepack as packer

HEADER_BYTES = 120
ENTRY_BYTES = 56
CHECK_TENSORS = (
    "transformer_blocks.0.attn.to_q.weight",
    "transformer_blocks.0.attn.to_k.weight",
    "transformer_blocks.0.attn.to_v.weight",
    "transformer_blocks.0.attn.to_out.0.weight",
    "transformer_blocks.49.attn.to_q.weight",
    "model.language_model.layers.0.self_attn.k_proj.weight",
    "model.language_model.layers.0.self_attn.v_proj.weight",
    "transformer_blocks.0.mlp.gate_proj.weight",
    "transformer_blocks.0.mlp.down_proj.weight",
)


def load_directory(path: Path) -> dict:
    with path.open("rb") as file:
        header = file.read(HEADER_BYTES)
        values = struct.unpack("<26IQQ", header)
        count = values[4]
        file.seek(values[26])
        entries = {}
        for _ in range(count):
            kind, layer, fmt, rows, columns, pad, offset, payload, so, sb = \
                struct.unpack("<IIIIIIQQQQ", file.read(ENTRY_BYTES))
            entries[(kind, layer)] = (rows, columns, offset, payload)
        return entries


def find_item(sections_inventory: dict, name: str) -> dict:
    for inventory in sections_inventory.values():
        for item in inventory:
            if item["name"] == name:
                return item
    raise SystemExit(f"{name} not found in warm inventory")


def warm_slice(warm: Path, item: dict, rank: int, tp16: dict) -> bytes:
    rows, columns = packer.flat_rows_columns(item["shape"])
    plan = packer.plan_of(item, 16)
    row_start, row_count, col_start, col_count = packer.tp_slice(
        rows, columns, plan, rank, 16, tp16)
    with (warm / packer.COMPONENT_DIRS[item["section"]] / item["shard"]).open("rb") as file:
        return packer.read_tensor_blob(file, item["shape"], item["dtype"],
                                       item["data_offsets"], row_start,
                                       row_count, col_start, col_count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, default=Path("/mnt/model-warm/minimax-h3"))
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--ranks", default="0,7,8,15")
    args = parser.parse_args()
    ranks = [int(part) for part in args.ranks.split(",")]
    spec = json.loads((ROOT / "model-families" / "minimax_h3" / "tensor_patterns.json").read_text())
    tp16 = spec["tp16"]
    codes = packer.assign_kind_codes(spec)
    sections_inventory = {}
    patterns = []
    for entry in spec["patterns"]:
        rx = entry["regex"].replace("{N}", "(\\d+)").replace("{M}", "(\\d+)")
        patterns.append(dict(entry, rx=re.compile(rx)))
    excluded = [re.compile(rx) for rx in spec.get("excluded", [])]
    for section in packer.SECTIONS:
        sections_inventory[section] = packer.component_inventory(
            section, args.warm, patterns, codes, excluded)
    q_code = codes[("dit", "DIT_ATTENTION_QUERY")]
    k_code = codes[("dit", "DIT_ATTENTION_KEY")]
    v_code = codes[("dit", "DIT_ATTENTION_VALUE")]
    o_code = codes[("dit", "DIT_ATTENTION_OUTPUT")]
    ek_code = codes[("encoder", "ENCODER_ATTENTION_KEY")]
    ev_code = codes[("encoder", "ENCODER_ATTENTION_VALUE")]
    gate_code = codes[("dit", "DIT_FFN_GATE_UP")]
    down_code = codes[("dit", "DIT_FFN_DOWN")]
    failures = []
    directories = {}
    for rank in ranks:
        directories[rank] = load_directory(args.packs / f"h3.bf16.tp16.rank{rank:02d}.sp")
        head_rows = tp16["dit_head_counts"][rank] * 128
        entries = directories[rank]
        for code in (q_code, k_code, v_code):
            if any(key[0] == code and entries[key][0] != head_rows for key in entries):
                failures.append(f"rank{rank:02d} dit q/k/v rows != {head_rows}")
        if any(key[0] == o_code and entries[key][1] != head_rows for key in entries):
            failures.append(f"rank{rank:02d} to_out columns != {head_rows}")
        for code in (ek_code, ev_code):
            if any(key[0] == code and entries[key][0] != 128 for key in entries):
                failures.append(f"rank{rank:02d} encoder k/v rows != 128")
        if any(key[0] == gate_code and entries[key][0] != 1792 for key in entries):
            failures.append(f"rank{rank:02d} ffn gate_up rows != 1792")
        if any(key[0] == down_code and entries[key][1] != 896 for key in entries):
            failures.append(f"rank{rank:02d} ffn down columns != 896")
        print(f"rank{rank:02d}: {len(entries)} entries, head rows {head_rows}, "
              f"kv head {rank // tp16['encoder_kv_replication']}")
    for rank in ranks:
        for name in CHECK_TENSORS:
            item = find_item(sections_inventory, name)
            rows, columns = packer.flat_rows_columns(item["shape"])
            plan = packer.plan_of(item, 16)
            rs, rc, cs, cc = packer.tp_slice(rows, columns, plan, rank, 16, tp16)
            payload_bytes = rc * cc * packer.DTYPE_BYTES[item["dtype"]]
            key = (item["code"], item["layer"])
            if key not in directories[rank]:
                failures.append(f"rank{rank:02d} {name}: directory entry absent")
                continue
            packed_rows, packed_cols, offset, payload = directories[rank][key]
            if (packed_rows, packed_cols, payload) != (rc, cc, payload_bytes):
                failures.append(f"rank{rank:02d} {name}: directory extent "
                                f"{(packed_rows, packed_cols, payload)} != "
                                f"{(rc, cc, payload_bytes)}")
                continue
            expected = warm_slice(args.warm, item, rank, tp16)
            with (args.packs / f"h3.bf16.tp16.rank{rank:02d}.sp").open("rb") as file:
                file.seek(offset)
                packed = file.read(payload)
            if hashlib.sha256(packed).hexdigest() != \
                    hashlib.sha256(expected).hexdigest():
                failures.append(f"rank{rank:02d} {name}: payload sha mismatch vs warm slice")
        state = "PASS" if not failures else "FAIL"
        print(f"{state} rank{rank:02d}: {len(CHECK_TENSORS)} payload shas vs warm slices")
    if 7 in directories and 8 in directories:
        concat_proof(args, sections_inventory, tp16, directories,
                     codes[("dit", "DIT_ATTENTION_QUERY")], failures)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"BOUNDARY CHECK FAIL ({len(failures)})")
        return 1
    print("BOUNDARY CHECK GREEN")
    return 0


def concat_proof(args, sections_inventory, tp16, directories, q_code, failures) -> None:
    item = find_item(sections_inventory, "transformer_blocks.0.attn.to_q.weight")
    blobs = {}
    for rank in (7, 8):
        rs, rc, cs, cc = packer.tp_slice(item_rows(item), item_cols(item),
                                         "heads_rows", rank, 16, tp16)
        key = (q_code, item["layer"])
        packed_rows, packed_cols, offset, payload = directories[rank][key]
        if (packed_rows, packed_cols) != (rc, cc):
            failures.append(f"concat: rank{rank:02d} extent "
                            f"{(packed_rows, packed_cols)} != {(rc, cc)}")
            return
        with (args.packs / f"h3.bf16.tp16.rank{rank:02d}.sp").open("rb") as file:
            file.seek(offset)
            blobs[rank] = file.read(payload)
    with (args.warm / "transformer" / item["shard"]).open("rb") as file:
        middle = packer.read_tensor_blob(file, item["shape"], item["dtype"],
                                         item["data_offsets"], 3584, 896)
    joined = blobs[7] + blobs[8]
    if hashlib.sha256(joined).hexdigest() != hashlib.sha256(middle).hexdigest():
        failures.append("concat: rank07+08 packed q bytes != warm [3584,4480)")
    else:
        print(f"PASS concat: rank07+08 packed q == warm rows [3584,4480) "
              f"({len(joined)} bytes)")


def item_rows(item: dict) -> int:
    return packer.flat_rows_columns(item["shape"])[0]


def item_cols(item: dict) -> int:
    return packer.flat_rows_columns(item["shape"])[1]


if __name__ == "__main__":
    sys.exit(main())
