#!/usr/bin/env python3
"""Convert the MiniMax-H3 text tower (MiniMaxH3ModularPipeline monorepo,
text_encoder/) into minimax stage packs.

Setup-time code, never the serving path: reads safetensors shard headers
and streams payloads into a dense-GQA wire format modeled on
tools/qwen38_27b_stagepack.py (120-byte 26I2Q header, 56-byte 6I4Q
entries, 256-byte payload alignment). The text tower is a Qwen3VL text
submodel per text_encoder/config.json: 64 dense layers, hidden 5120,
64 query heads x 128, 8 KV heads, FFN 25600, vocab 151936, RMSNorm
without the +1 fold, untied lm_head, no biases, no MTP, all BF16.

The vision tower (model.visual.*), transformer/, audio_* and FL2VA are
out of scope per the merged #1044 serving target and are never
referenced; every consumed text-tower tensor is pinned by name, dtype,
and shape, and the config pin fails closed on any other release.

Topology: TP4 is the emitted arm (minimax.text.bf16.tp4). The 5120/64L
geometry is the qwen38_27b family shape whose TPmax is TP4; all shard
dimensions divide (q heads 64/4, kv heads 8/4 >= 1 so GQA shards without
replication, FFN 25600/4, vocab 151936/4). The embedding replicates on
every rank (no collective broadcast yet; the gather path reads the full
table on every rank), lm_head shards by vocab rows.

Per-rank emission is a two-pass proof: the pack is written, then
--verify (or the automatic tail call) re-parses the file and re-derives
every directory entry from the inventory. No module consumes this wire
yet; the pack is the serving-target artifact for the minimax lane.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from spark_pack_common import (  # noqa: E402
    PackFailure,
    SafetensorsSource,
    align_up,
    sha256_file,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]

MAGIC = 0x58544E4D  # 'MNTX' little endian
FORMAT_VERSION = 1
HEADER_BYTES = 120
ENTRY_BYTES = 56
GLOBAL_LAYER = 0xFFFFFFFF
PAYLOAD_ALIGNMENT = 256
WEIGHT_BF16 = 0
BF16_BYTES = 2

HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES

HIDDEN = 5120
LAYER_COUNT = 64
QUERY_HEADS = 64
KV_HEADS = 8
HEAD_DIM = 128
FFN_INTERMEDIATE = 25600
VOCAB = 151936
Q_ROWS = QUERY_HEADS * HEAD_DIM          # 8192
KV_ROWS = KV_HEADS * HEAD_DIM            # 1024
O_COLS = QUERY_HEADS * HEAD_DIM          # 8192

(KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD, KIND_INPUT_NORM,
 KIND_POST_ATTENTION_NORM, KIND_Q_NORM, KIND_K_NORM, KIND_QUERY,
 KIND_KEY, KIND_VALUE, KIND_OUTPUT, KIND_FFN_GATE, KIND_FFN_UP,
 KIND_FFN_DOWN) = range(14)

ROW_SHARDED_KINDS = frozenset([KIND_QUERY, KIND_KEY, KIND_VALUE,
                               KIND_FFN_GATE, KIND_FFN_UP])
COL_SHARDED_KINDS = frozenset([KIND_OUTPUT, KIND_FFN_DOWN])
REPLICATED_KINDS = frozenset([KIND_EMBEDDING, KIND_FINAL_NORM,
                              KIND_INPUT_NORM, KIND_POST_ATTENTION_NORM,
                              KIND_Q_NORM, KIND_K_NORM])
GLOBAL_KINDS = frozenset([KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD])

KIND_TENSORS = {
    KIND_INPUT_NORM: "input_layernorm.weight",
    KIND_POST_ATTENTION_NORM: "post_attention_layernorm.weight",
    KIND_Q_NORM: "self_attn.q_norm.weight",
    KIND_K_NORM: "self_attn.k_norm.weight",
    KIND_QUERY: "self_attn.q_proj.weight",
    KIND_KEY: "self_attn.k_proj.weight",
    KIND_VALUE: "self_attn.v_proj.weight",
    KIND_OUTPUT: "self_attn.o_proj.weight",
    KIND_FFN_GATE: "mlp.gate_proj.weight",
    KIND_FFN_UP: "mlp.up_proj.weight",
    KIND_FFN_DOWN: "mlp.down_proj.weight",
}

PREFIX = "model.language_model."

CHUNK_BYTES = 8 * 1024 * 1024


def kind_shape(kind: int) -> tuple[int, int, int]:
    table = {
        KIND_EMBEDDING: (VOCAB, HIDDEN),
        KIND_FINAL_NORM: (1, HIDDEN),
        KIND_LM_HEAD: (VOCAB, HIDDEN),
        KIND_INPUT_NORM: (1, HIDDEN),
        KIND_POST_ATTENTION_NORM: (1, HIDDEN),
        KIND_Q_NORM: (1, HEAD_DIM),
        KIND_K_NORM: (1, HEAD_DIM),
        KIND_QUERY: (Q_ROWS, HIDDEN),
        KIND_KEY: (KV_ROWS, HIDDEN),
        KIND_VALUE: (KV_ROWS, HIDDEN),
        KIND_OUTPUT: (HIDDEN, O_COLS),
        KIND_FFN_GATE: (FFN_INTERMEDIATE, HIDDEN),
        KIND_FFN_UP: (FFN_INTERMEDIATE, HIDDEN),
        KIND_FFN_DOWN: (HIDDEN, FFN_INTERMEDIATE),
    }
    rows, columns = table[kind]
    return rows, columns, WEIGHT_BF16


def build_inventory(tp_degree: int, tp_rank: int) -> list[dict]:
    if VOCAB % tp_degree or Q_ROWS % tp_degree or KV_ROWS % tp_degree \
            or FFN_INTERMEDIATE % tp_degree:
        raise PackFailure(f"tp degree {tp_degree} does not divide the shard dimensions")
    plan = []
    for kind in (KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD):
        rows, columns, _ = kind_shape(kind)
        if kind == KIND_LM_HEAD:
            rows = rows // tp_degree
            row_base = tp_rank * rows
        else:
            row_base = 0
        plan.append(dict(kind=kind, layer=GLOBAL_LAYER, rows=rows, columns=columns,
                         weight_format=WEIGHT_BF16, row_base=row_base,
                         col_base=0, source="global"))
    for layer in range(LAYER_COUNT):
        for kind in (KIND_INPUT_NORM, KIND_POST_ATTENTION_NORM, KIND_Q_NORM,
                     KIND_K_NORM, KIND_QUERY, KIND_KEY, KIND_VALUE,
                     KIND_OUTPUT, KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN):
            rows, columns, _ = kind_shape(kind)
            row_base = col_base = 0
            if kind in ROW_SHARDED_KINDS:
                rows = rows // tp_degree
                row_base = tp_rank * rows
            elif kind in COL_SHARDED_KINDS:
                columns = columns // tp_degree
                col_base = tp_rank * columns
            plan.append(dict(kind=kind, layer=layer, rows=rows, columns=columns,
                             weight_format=WEIGHT_BF16, row_base=row_base,
                             col_base=col_base, source="layer"))
    return plan


def tensor_name(entry: dict) -> str:
    if entry["source"] == "global":
        if entry["kind"] == KIND_EMBEDDING:
            return PREFIX + "embed_tokens.weight"
        if entry["kind"] == KIND_FINAL_NORM:
            return PREFIX + "norm.weight"
        return "lm_head.weight"
    return (PREFIX + f"layers.{entry['layer']}."
            + KIND_TENSORS[entry["kind"]])


def consumed_names() -> set[str]:
    names = {PREFIX + "embed_tokens.weight", PREFIX + "norm.weight",
             "lm_head.weight"}
    for kind in KIND_TENSORS:
        names.add(PREFIX + f"layers.0.{KIND_TENSORS[kind]}")
    return names


def check_source(source: SafetensorsSource) -> None:
    text = source.config.get("text_config", {})
    expectations = {
        "hidden_size": HIDDEN, "num_hidden_layers": LAYER_COUNT,
        "num_attention_heads": QUERY_HEADS, "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM, "intermediate_size": FFN_INTERMEDIATE,
        "vocab_size": VOCAB,
    }
    for key, expected in expectations.items():
        if text.get(key) != expected:
            raise PackFailure(f"config.json text_config.{key}={text.get(key)!r}, "
                              f"expected {expected!r}")
    if source.config.get("tie_word_embeddings") not in (False, None):
        raise PackFailure("config.json tie_word_embeddings is not false; the "
                          "pack carries a separate lm_head slot")
    if source.config.get("quantization_config"):
        raise PackFailure("source carries quantization_config; the text tower "
                          "packer takes the BF16 release only")
    for name in sorted(consumed_names()):
        shard, meta, offset = source.resolve(name)
        if meta["dtype"] != "BF16":
            raise PackFailure(f"{name}: dtype {meta['dtype']}, expected BF16")


def check_entry_shape(source: SafetensorsSource, entry: dict) -> None:
    name = tensor_name(entry)
    _, meta, _ = source.resolve(name)
    full_rows, full_columns, _ = kind_shape(entry["kind"])
    shape = list(meta["shape"])
    if len(shape) == 1:
        shape = [1, shape[0]]
    if shape != [full_rows, full_columns]:
        raise PackFailure(f"{name}: checkpoint shape {meta['shape']}, pack expects "
                          f"[{full_rows}, {full_columns}]")


def copy_entry(source: SafetensorsSource, entry: dict, out, offset: int) -> None:
    name = tensor_name(entry)
    path = source.root / source.weight_map[name]
    _, _, base = source.resolve(name)
    element = BF16_BYTES
    rows, columns = entry["rows"], entry["columns"]
    full_rows, full_columns, _ = kind_shape(entry["kind"])
    with open(path, "rb") as file:
        row_bytes = full_columns * element
        if entry["row_base"] == 0 and entry["col_base"] == 0 \
                and rows == full_rows and columns == full_columns:
            file.seek(base)
            want = rows * columns * element
            remaining = want
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                chunk = file.read(step)
                if len(chunk) != step:
                    raise PackFailure(f"{name}: short read")
                out.write(chunk)
                remaining -= step
            return
        if entry["col_base"] == 0 and columns == full_columns:
            file.seek(base + entry["row_base"] * row_bytes)
            want = rows * row_bytes
            remaining = want
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                chunk = file.read(step)
                if len(chunk) != step:
                    raise PackFailure(f"{name}: short read")
                out.write(chunk)
                remaining -= step
            return
        step_rows = max(1, CHUNK_BYTES // row_bytes)
        if columns != full_columns:
            step_rows = 1
        row = 0
        while row < rows:
            block = min(step_rows, rows - row)
            file.seek(base + (entry["row_base"] + row) * row_bytes
                      + entry["col_base"] * element)
            want = block * columns * element
            remaining = want
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                chunk = file.read(step)
                if len(chunk) != step:
                    raise PackFailure(f"{name}: short read")
                out.write(chunk)
                remaining -= step
            row += block


def convert(checkpoint: Path, output: Path, tp_degree: int, tp_rank: int,
            receipt: dict, dry_run: bool) -> dict:
    source = SafetensorsSource(checkpoint)
    check_source(source)
    plan = build_inventory(tp_degree, tp_rank)
    checked = set()
    for entry in plan:
        name = tensor_name(entry)
        if name not in checked:
            check_entry_shape(source, entry)
            checked.add(name)
    cursor = HEADER_BYTES + len(plan) * ENTRY_BYTES
    payload_base = align_up(cursor, PAYLOAD_ALIGNMENT)
    entries = []
    offset = payload_base
    for entry in plan:
        entry["payload_offset"] = offset
        entry["payload_bytes"] = entry["rows"] * entry["columns"] * BF16_BYTES
        offset = align_up(offset + entry["payload_bytes"], PAYLOAD_ALIGNMENT)
        entries.append(entry)
    file_bytes = entries[-1]["payload_offset"] + entries[-1]["payload_bytes"]
    header = HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(plan),
        HIDDEN, LAYER_COUNT, 0, LAYER_COUNT,
        QUERY_HEADS, KV_HEADS, HEAD_DIM, FFN_INTERMEDIATE, VOCAB,
        0, tp_degree, tp_rank, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        HEADER_BYTES, file_bytes)
    receipt.update({
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "tensor_count": len(plan),
        "bytes": file_bytes,
        "source_index_sha256": source.index_sha256,
        "source_config_sha256": source.config_sha256,
    })
    if dry_run:
        print(f"minimax_stagepack tp{tp_degree} rank {tp_rank} "
              f"tensors={len(plan)} file_gib={file_bytes / 2**30:.2f} (dry run)")
        return receipt

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp",
                                     dir=output.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        temp.write(header)
        for entry in entries:
            temp.write(ENTRY_STRUCT.pack(
                entry["kind"], entry["layer"], entry["weight_format"],
                entry["rows"], entry["columns"], 0,
                entry["payload_offset"], entry["payload_bytes"], 0, 0))
        padding = payload_base - temp.tell()
        if padding < 0:
            raise PackFailure("directory overruns the payload base")
        temp.write(b"\0" * padding)
        for entry in entries:
            before = temp.tell()
            copy_entry(source, entry, temp, entry["payload_offset"])
            if temp.tell() - before != entry["payload_bytes"]:
                raise PackFailure(f"payload size mismatch on {tensor_name(entry)}")
            pad = align_up(temp.tell(), PAYLOAD_ALIGNMENT) - temp.tell()
            if pad:
                temp.write(b"\0" * pad)
        temp.truncate(file_bytes)
        temp.flush()
        os.fsync(temp.fileno())
    os.replace(temp_path, output)
    receipt["output_sha256"] = sha256_file(output)
    receipt["file"] = str(output)
    proof = verify(output)
    if proof["tensor_count"] != len(plan) or not proof["passed"]:
        raise PackFailure("tail verify failed on the freshly written pack")
    receipt["verify_proof"] = proof
    print(f"minimax_stagepack tp{tp_degree} rank {tp_rank} tensors={len(plan)} "
          f"file_gib={file_bytes / 2**30:.2f} wrote {output} verify=PASS")
    return receipt


def verify(pack_path: Path, tp_degree: int = 0) -> dict:
    """Re-parse a pack and check every wire rule; also re-derives the
    inventory census from the geometry fields."""
    file_bytes = pack_path.stat().st_size
    with pack_path.open("rb") as file:
        raw_header = file.read(HEADER_BYTES)
        if len(raw_header) != HEADER_BYTES:
            raise PackFailure("short header")
        fields = HEADER_STRUCT.unpack(raw_header)
        (magic, version, header_bytes, entry_bytes, tensor_count, hidden,
         layer_count, first_layer, total_layers, q_heads, kv_heads, head_dim,
         ffn, vocab, mtp, tp_degree_field, tp_rank_field, reserved0,
         reserved1, reserved2, reserved3, reserved4, reserved5, reserved6,
         reserved7, reserved8, directory_offset, declared_bytes) = fields
        geometry = {
            "magic": (magic, MAGIC), "format_version": (version, FORMAT_VERSION),
            "header_bytes": (header_bytes, HEADER_BYTES),
            "directory_entry_bytes": (entry_bytes, ENTRY_BYTES),
            "hidden_dimension": (hidden, HIDDEN),
            "layer_count": (layer_count, layer_count),
            "first_layer_index": (first_layer, 0),
            "total_layer_count": (total_layers, LAYER_COUNT),
            "query_head_count": (q_heads, QUERY_HEADS),
            "kv_head_count": (kv_heads, KV_HEADS),
            "head_dimension": (head_dim, HEAD_DIM),
            "ffn_intermediate_dimension": (ffn, FFN_INTERMEDIATE),
            "output_vocab_count": (vocab, VOCAB),
            "mtp_layer_count": (mtp, 0),
        }
        for name, (actual, expected) in geometry.items():
            if actual != expected:
                raise PackFailure(f"geometry field {name}: {actual}, expected {expected}")
        if any((reserved0, reserved1, reserved2, reserved3, reserved4,
                reserved5, reserved6, reserved7, reserved8)):
            raise PackFailure("reserved header fields must be zero")
        if directory_offset != HEADER_BYTES or declared_bytes != file_bytes:
            raise PackFailure("directory offset or file size mismatch")
        if tp_degree_field < 1 or tp_rank_field >= tp_degree_field:
            raise PackFailure(f"tp fields {tp_degree_field}/{tp_rank_field} invalid")
        if tp_degree and (tp_degree_field != tp_degree or tp_rank_field >= tp_degree):
            raise PackFailure(f"tp fields {tp_degree_field}/{tp_rank_field} "
                              f"conflict with the expected degree {tp_degree}")
        expected_plan_shape = build_inventory(tp_degree_field, tp_rank_field)
        if tensor_count != len(expected_plan_shape):
            raise PackFailure(f"tensor_count {tensor_count}, inventory expects "
                              f"{len(expected_plan_shape)}")
        raw_directory = file.read(tensor_count * ENTRY_BYTES)
        if len(raw_directory) != tensor_count * ENTRY_BYTES:
            raise PackFailure("short directory")
        payload_base = align_up(HEADER_BYTES + tensor_count * ENTRY_BYTES,
                                PAYLOAD_ALIGNMENT)
        seen = set()
        cursor = payload_base
        for index in range(tensor_count):
            entry = ENTRY_STRUCT.unpack_from(raw_directory, index * ENTRY_BYTES)
            (kind, layer, weight_format, rows, columns, scale_group,
             payload_offset, payload_bytes, scale_offset, scale_bytes) = entry
            expected = expected_plan_shape[index]
            if (kind, layer) != (expected["kind"], expected["layer"]):
                raise PackFailure(f"entry {index}: kind/layer {(kind, layer)} != "
                                  f"{(expected['kind'], expected['layer'])}")
            if (rows, columns) != (expected["rows"], expected["columns"]):
                raise PackFailure(f"entry {index} kind {kind}: shape mismatch")
            if weight_format != WEIGHT_BF16:
                raise PackFailure(f"entry {index} kind {kind}: only BF16 is wired")
            if scale_group or scale_offset or scale_bytes:
                raise PackFailure(f"entry {index} kind {kind}: BF16 carries no scales")
            if (kind, layer) in seen:
                raise PackFailure(f"duplicate tensor kind {kind} layer {layer}")
            seen.add((kind, layer))
            if payload_offset != cursor or payload_offset % PAYLOAD_ALIGNMENT != 0:
                raise PackFailure(f"entry {index}: payload offset {payload_offset}, "
                                  f"expected {cursor}")
            if payload_bytes != rows * columns * BF16_BYTES:
                raise PackFailure(f"entry {index}: payload byte count mismatch")
            cursor = align_up(payload_offset + payload_bytes, PAYLOAD_ALIGNMENT)
        if cursor > file_bytes:
            raise PackFailure("trailing payload does not close the file")
    return {"file": str(pack_path), "bytes": file_bytes,
            "tensor_count": tensor_count, "tp_degree": tp_degree_field,
            "tp_rank": tp_rank_field, "passed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path,
                        help="text_encoder directory of the warm monorepo")
    parser.add_argument("--output", type=Path, help="pack output path")
    parser.add_argument("--receipt", type=Path,
                        help="receipt output (default: <output>.receipt.json)")
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-all", type=int, default=0,
                        help="emit all N rank packs in one process")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", type=Path, help="verify an existing pack and exit")
    args = parser.parse_args()

    if args.verify is not None:
        result = verify(args.verify)
        print(f"minimax_stagepack verify ok: {result['file']} "
              f"tensors={result['tensor_count']} "
              f"tp{result['tp_degree']}/{result['tp_rank']} bytes={result['bytes']}")
        return 0

    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    if args.output is None and not (args.dry_run or args.tp_all):
        parser.error("--output is required unless --dry-run or --tp-all")
    if args.tp_degree < 1 or args.tp_rank < 0 or args.tp_rank >= args.tp_degree:
        parser.error("--tp-rank must satisfy 0 <= tp-rank < tp-degree")

    ranks = range(args.tp_all) if args.tp_all else [args.tp_rank]
    for rank in ranks:
        receipt = {
            "kind": "sparkpipe.minimax.stagepack-receipt.v1",
            "tool": "tools/minimax_stagepack.py",
            "checkpoint": str(args.checkpoint),
        }
        if args.tp_all:
            out_dir = args.output or (args.checkpoint.parent / "packs")
            output = Path(out_dir) / (f"minimax.text.bf16.tp{args.tp_degree}"
                                      f".rank{rank:x}.mntx")
        else:
            output = args.output
        result = convert(args.checkpoint, output, args.tp_degree, rank,
                         receipt, args.dry_run)
        if not args.dry_run:
            receipt_path = Path(str(output) + ".receipt.json")
            write_receipt(result, receipt_path, suffix=None)
            sha_path = Path(str(output) + ".sha256")
            sha_path.write_text(f"{result['output_sha256']}  {output.name}\n")
            print(f"minimax_stagepack receipt {receipt_path} sha {sha_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"minimax_stagepack: {error}", file=sys.stderr)
        sys.exit(1)
