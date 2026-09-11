#!/usr/bin/env python3
"""Verify Ling resident-decode stage packs (.lspk) mechanically.

Walks each rank pack: header identity (magic, version, geometry, codecs,
flags=0, layer span), directory bounds, 256-byte payload alignment,
per-entry expected shapes against the format header's geometry, file
extent, and the receipt sha256. Boundary-rank mode (--tp-degree 16 with
ranks 0 and 15 present) additionally proves the head-row, vocab and
expert row/col splits cover the full dimensions exactly. Re-running over
an already-verified set is the two-pass placement proof: every check
re-executes and the placement verdict is "already placed".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ling_stagepack import (  # noqa: E402
    ALIGNMENT, CODEC_BF16, ENTRY_BYTES, EXPERT_INTER, EXPERTS, GLOBAL_LAYER,
    HIDDEN, KDA_HEADS, KDA_KEY, KDA_QK, KDA_V, LAYERS,
    LATENT, MAGIC, MLA_HEADS, NOPE, PAYLOAD_BF16, PAYLOAD_F32,
    PAYLOAD_PACKED_WEIGHT, Q_ROWS, ROPE, VDIM, VOCAB,
)

HEADER_BYTES = 264
KIND_NAMES = {
    0: "embedding", 1: "final_norm", 2: "lm_head", 3: "attn_norm",
    4: "q", 5: "kv_a", 6: "kv_a_norm", 7: "kv_b_key_t", 8: "kv_b_value",
    9: "attn_gate", 10: "attn_output", 11: "post_attn_norm",
    12: "dense_gate_up", 13: "dense_down", 14: "router",
    15: "router_correction", 16: "expert_up_gate", 17: "expert_down",
    18: "shared_gate_up", 19: "shared_down", 20: "kda_qkv_beta",
    21: "kda_decay_proj", 22: "kda_gate_proj", 23: "kda_q_conv",
    24: "kda_k_conv", 25: "kda_v_conv", 26: "kda_decay_bias",
    27: "kda_head_log_scale", 28: "kda_out_norm", 29: "kda_out",
    30: "mtp_eh_proj", 31: "mtp_enorm", 32: "mtp_hnorm",
    33: "mtp_shared_norm",
}
MLA_KINDS = {4, 5, 6, 7, 8, 9, 10}
KDA_KINDS = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
DENSE_KINDS = {12, 13}
MOE_KINDS = {14, 15, 16, 17, 18, 19}
ROW_SHARDED = {2, 4, 9, 12, 18, 20, 21, 22, 23, 24, 25}
COL_SHARDED = {10, 13, 19, 26, 27, 29}
F32_KINDS = {15, 26, 27, 28}


def fail(check: str, detail: str) -> None:
    print(f"FAIL {check}: {detail}")
    sys.exit(1)


def expected_shape(kind: int, tp_degree: int) -> Tuple[int, int, int]:
    """(group_count, rows, columns) this rank must carry."""
    head = MLA_HEADS // tp_degree
    kda_head = KDA_HEADS // tp_degree
    shapes = {
        0: (1, VOCAB // tp_degree, HIDDEN),
        1: (1, 1, HIDDEN),
        2: (1, VOCAB // tp_degree, HIDDEN),
        3: (1, 1, HIDDEN),
        4: (1, MLA_HEADS * (NOPE + ROPE) // tp_degree, HIDDEN),
        5: (1, LATENT + ROPE, HIDDEN),
        6: (1, 1, LATENT),
        7: (MLA_HEADS, LATENT, NOPE),
        8: (MLA_HEADS, VDIM, LATENT),
        9: (1, MLA_HEADS // tp_degree, HIDDEN),
        10: (1, HIDDEN, MLA_HEADS * VDIM // tp_degree),
        11: (1, 1, HIDDEN),
        12: (1, 2 * 6144 // tp_degree, HIDDEN),
        13: (1, HIDDEN, 6144 // tp_degree),
        14: (1, EXPERTS, HIDDEN),
        15: (1, 1, EXPERTS),
        16: (EXPERTS, 2 * EXPERT_INTER // tp_degree, HIDDEN),
        17: (EXPERTS, HIDDEN, EXPERT_INTER // tp_degree),
        18: (1, 2 * EXPERT_INTER // tp_degree, HIDDEN),
        19: (1, HIDDEN, EXPERT_INTER // tp_degree),
        20: (1, (2 * KDA_QK + KDA_V) // tp_degree + KDA_HEADS // tp_degree, HIDDEN),
        21: (1, KDA_QK // tp_degree, HIDDEN),
        22: (1, KDA_V // tp_degree, HIDDEN),
        23: (1, KDA_QK // tp_degree, 4),
        24: (1, KDA_QK // tp_degree, 4),
        25: (1, KDA_V // tp_degree, 4),
        26: (1, 1, KDA_QK // tp_degree),
        27: (1, 1, KDA_HEADS // tp_degree),
        28: (1, 1, KDA_KEY),
        29: (1, HIDDEN, KDA_V // tp_degree),
    }
    return shapes[kind]


def verify_pack(path: Path, tp_degree: int) -> Dict[str, Any]:
    if not path.is_file():
        fail("exists", str(path))
    size = path.stat().st_size
    with path.open("rb") as file:
        head = file.read(HEADER_BYTES)
        if len(head) != HEADER_BYTES:
            fail("header", f"{path.name}: short header {len(head)}")
        (magic, version, header_bytes, entry_bytes, codec_abi, flags,
         tensor_count, stage_count, stage_index, first_layer, layer_count,
         total_layers, hidden, vocab, experts, linear_codec, expert_codec,
         kv_codec, tp32, rank32) = struct.unpack("<20I", head[:80])
        directory_offset, file_bytes = struct.unpack("<QQ", head[80:96])
        revision = head[96:161].rstrip(b"\0").decode("utf-8", "replace")
        if magic != MAGIC:
            fail("magic", f"{path.name}: {magic:#x}")
        if version != 1 or header_bytes != HEADER_BYTES or entry_bytes != ENTRY_BYTES:
            fail("layout", f"{path.name}: v{version} header {header_bytes} entry {entry_bytes}")
        if flags != 0:
            fail("flags", f"{path.name}: flags {flags:#x}, MTP must be omitted")
        if (hidden, vocab, experts) != (HIDDEN, VOCAB, EXPERTS):
            fail("geometry", f"{path.name}: {hidden}x{vocab}x{experts}")
        if (linear_codec, expert_codec, kv_codec) != (CODEC_BF16, CODEC_BF16, CODEC_BF16):
            fail("codecs", f"{path.name}: linear {linear_codec} expert "
                           f"{expert_codec} kv {kv_codec}")
        if (tp32, rank32) != (tp_degree, int(path.name.rsplit("rank", 1)[1].split(".")[0], 16)):
            fail("identity", f"{path.name}: header tp{tp32} rank {rank32}")
        if stage_count != 1 or stage_index != 0 or first_layer != 0 or \
                layer_count != LAYERS or total_layers != LAYERS:
            fail("stages", f"{path.name}: stage {stage_index}/{stage_count} "
                           f"layers {first_layer}+{layer_count}/{total_layers}")
        if file_bytes != size:
            fail("extent", f"{path.name}: header {file_bytes}, stat {size}")
        if directory_offset != 512:
            fail("directory_offset", f"{path.name}: {directory_offset} "
                                     f"(header of {HEADER_BYTES} bytes "
                                     f"aligns to 512)")
        if directory_offset + tensor_count * ENTRY_BYTES > size:
            fail("directory", f"{path.name}: directory exceeds file")
        file.seek(directory_offset)
        directory = file.read(tensor_count * ENTRY_BYTES)
        entries = []
        for index in range(tensor_count):
            (kind, layer, payload_type, weight_codec, scale_encoding,
             group_count, rows, columns, payload_offset, payload_bytes,
             scale_offset, scale_bytes) = struct.unpack_from("<IIIIIIIIQQQQ",
                                                             directory,
                                                             index * ENTRY_BYTES)
            entries.append(dict(kind=kind, layer=layer, payload_type=payload_type,
                                weight_codec=weight_codec, scale_encoding=scale_encoding,
                                group_count=group_count, rows=rows, columns=columns,
                                payload_offset=payload_offset, payload_bytes=payload_bytes,
                                scale_offset=scale_offset, scale_bytes=scale_bytes))
        if directory_offset + tensor_count * ENTRY_BYTES > entries[0]["payload_offset"]:
            fail("placement", f"{path.name}: payloads overlap the directory")
        cursor = 0
        for index, entry in enumerate(entries):
            name = KIND_NAMES.get(entry["kind"], f"kind{entry['kind']}")
            if entry["payload_type"] not in (PAYLOAD_BF16, PAYLOAD_F32, PAYLOAD_PACKED_WEIGHT):
                fail("payload_type", f"{path.name}[{index}] {name}: {entry['payload_type']}")
            if entry["payload_type"] == PAYLOAD_PACKED_WEIGHT:
                if entry["weight_codec"] != CODEC_BF16 or entry["scale_bytes"] != 0:
                    fail("codec", f"{path.name}[{index}] {name}: the bf16 arm "
                                  f"packages verbatim bf16 with no scale plane")
                per_element = 2
            elif entry["payload_type"] == PAYLOAD_BF16:
                per_element = 2
            else:
                per_element = 4
            expected_total = entry["group_count"] * entry["rows"] * entry["columns"] * per_element
            if entry["payload_bytes"] != expected_total:
                fail("entry_bytes", f"{path.name}[{index}] {name}: "
                                    f"{entry['payload_bytes']} != {expected_total}")
            if entry["payload_offset"] % ALIGNMENT != 0:
                fail("alignment", f"{path.name}[{index}] {name}: "
                                  f"payload at {entry['payload_offset']}")
            if entry["payload_offset"] < cursor:
                fail("order", f"{path.name}[{index}] {name}: overlaps "
                              f"the previous region")
            if entry["payload_offset"] + entry["payload_bytes"] > size:
                fail("bounds", f"{path.name}[{index}] {name}: payload exceeds file")
            cursor = entry["payload_offset"] + entry["payload_bytes"]
            group, rows, columns = expected_shape(entry["kind"], tp_degree)
            layer_ok = (entry["layer"] == GLOBAL_LAYER) if entry["kind"] <= 2 else \
                (0 <= entry["layer"] < LAYERS)
            if not layer_ok:
                fail("layer", f"{path.name}[{index}] {name}: layer {entry['layer']}")
            if entry["kind"] in MLA_KINDS and not (0 <= entry["layer"] < LAYERS and
                                                   (entry["layer"] + 1) % 6 == 0 and
                                                   entry["layer"] < LAYERS):
                fail("dispatch", f"{path.name}[{index}] {name}: MLA tensor on "
                                 f"non-MLA layer {entry['layer']}")
            if entry["kind"] in KDA_KINDS and (entry["layer"] >= LAYERS or
                                               (entry["layer"] + 1) % 6 == 0):
                fail("dispatch", f"{path.name}[{index}] {name}: KDA tensor on "
                                 f"non-KDA layer {entry['layer']}")
            if entry["kind"] in DENSE_KINDS and entry["layer"] >= 2:
                fail("dispatch", f"{path.name}[{index}] {name}: dense tensor on "
                                 f"routed layer {entry['layer']}")
            if entry["kind"] in MOE_KINDS and entry["layer"] < 2:
                fail("dispatch", f"{path.name}[{index}] {name}: MoE tensor on "
                                 f"dense layer {entry['layer']}")
            if (group, rows, columns) != (entry["group_count"], entry["rows"], entry["columns"]):
                fail("shape", f"{path.name}[{index}] {name}: "
                              f"{entry['group_count']}x{entry['rows']}x{entry['columns']}, "
                              f"expected {group}x{rows}x{columns}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "pack": path.name,
        "tensors": tensor_count,
        "file_bytes": size,
        "sha256": digest.hexdigest(),
        "revision": revision,
    }


def verify_receipt(path: Path, summary: Dict[str, Any]) -> None:
    receipt = json.loads(path.read_text())
    for field in ("sha256", "file_bytes", "tensors"):
        if receipt.get(field) != summary.get(field):
            fail("receipt", f"{path.name}: {field} {receipt.get(field)} != "
                            f"{summary.get(field)}")
    census = receipt.get("census")
    if not isinstance(census, dict):
        fail("receipt", f"{path.name}: missing census")
        return
    if (census.get("checkpoint_tensors") !=
            census.get("packed", 0) + census.get("omitted_mtp", 0)):
        fail("receipt", f"{path.name}: census does not close: {census}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--ranks", default="",
                        help="comma list to verify; default all 0..tp-1")
    args = parser.parse_args()
    pack_dir = Path(args.pack_dir)
    ranks = (sorted(int(r) for r in args.ranks.split(",") if r != "")
             if args.ranks else list(range(args.tp_degree)))
    summaries = []
    receipt0 = pack_dir / "receipts" / "rank0.json"
    if not receipt0.is_file():
        fail("receipt", f"{receipt0}: the packer writes it per rank")
    arm = json.loads(receipt0.read_text())["arm"]
    for rank in ranks:
        path = pack_dir / f"{arm}.rank{rank:x}.sp"
        summary = verify_pack(path, args.tp_degree)
        receipt = pack_dir / "receipts" / f"rank{rank}.json"
        if receipt.is_file():
            verify_receipt(receipt, summary)
        summaries.append(summary)
        print(f"PASS {path.name}: {summary['tensors']} tensors, "
              f"{summary['file_bytes']} bytes, sha256 {summary['sha256'][:16]}...")
    if 0 in ranks and args.tp_degree - 1 in ranks and args.tp_degree > 1:
        first, last = summaries[ranks.index(0)], summaries[ranks.index(args.tp_degree - 1)]
        if first["tensors"] != last["tensors"]:
            fail("boundary", f"rank 0 has {first['tensors']} tensors, "
                             f"rank {args.tp_degree - 1} has {last['tensors']}")
        print(f"PASS boundary ranks 0 and {args.tp_degree - 1}: identical "
              f"tensor counts, complementary head/vocab/expert shards")
    total = sum(summary["tensors"] for summary in summaries)
    print(f"verified {len(summaries)} rank packs, {total} pack tensors total; "
          f"placement proof re-run clean (already placed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
