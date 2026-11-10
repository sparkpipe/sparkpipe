#!/usr/bin/env python3
"""Convert the meta-models/Muse-Glimmer-30B BF16 safetensors checkpoint into
muse_glimmer stage packs (lane muse).

Setup-time code, never the serving path. Built on tools/spark_pack_common.py
(index/header resolution, pread slice reads); the largest in-memory gather is
one fused qgkv plane (~10 MiB), so node RSS stays far under 1 GiB. Muse is
the campaign's first dense GQA family: no GDN, no MoE, no MTP - 12 kept
per-layer tensors + 3 globals, one EVERY_LAYER stagepack class.

TP plan (DESIGN section 5, asserted here):
  * q heads: rank r owns global heads 2r and 2r+1 (32 heads / 16 ranks).
  * kv heads: rank r owns kv head r//8 - ranks 0-7 replicate kv head 0,
    ranks 8-15 kv head 1; the 8 sharers must produce bitwise-identical K/V
    (post-pack anchor A7 checks the shards, not this tool).
  * fused qgkv rows per rank: [q(h0)|gate(h0)|q(h1)|gate(h1)|k|v] - the
    head-major q|gate interleave the shared LmSplitQueryGateKernel expects;
    gate(h) is the attention OUTPUT gate (elementwise sigmoid over the head).
  * o_proj columns [2r*128, 2r*128+256); mlp gate/up rows shards of 1248
    (gate half then up half); down K columns 1248; embedding + lm_head
    vocab rows 12628 at offset r*12628; the four norms and final norm
    replicate on every rank.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_codecs import WEIGHT_BF16  # noqa: E402
from spark_pack_common import (GLOBAL_LAYER, PackFailure, SafetensorsSource, align_up, sha256_file,  # noqa: E402
                               window_chunks, write_receipt)
from spark_pack_wire import PAYLOAD_ALIGNMENT, STAGEPACK_ENTRY, STAGEPACK_HEADER, region_offsets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERNS = ROOT / "model-families" / "muse_glimmer" / "tensor_patterns.json"

MAGIC = 0x47534D55
FORMAT_VERSION = 1
HEADER_BYTES = STAGEPACK_HEADER.size
ENTRY_BYTES = STAGEPACK_ENTRY.size
HEADER_STRUCT = STAGEPACK_HEADER
ENTRY_STRUCT = STAGEPACK_ENTRY

HIDDEN = 6656
LAYER_COUNT = 52
VOCAB = 202048
ATTN_QUERY_HEADS = 32
ATTN_KV_HEADS = 2
ATTN_HEAD_DIM = 128
INTERMEDIATE = 19968
ATTENTION_PERIOD = 4
FULL_PHASE = 3

LAYER_PREFIX = "model.language_model.layers."

(KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD, KIND_INPUT_NORM,
 KIND_POST_ATTENTION_NORM, KIND_PRE_FFN_NORM, KIND_POST_FFN_NORM,
 KIND_QGKV, KIND_ATTN_OUTPUT, KIND_ATTN_GATE_UP, KIND_MLP_DOWN) = range(11)

KIND_NAMES = {
    KIND_EMBEDDING: "EMBEDDING",
    KIND_FINAL_NORM: "FINAL_NORM",
    KIND_LM_HEAD: "LM_HEAD",
    KIND_INPUT_NORM: "INPUT_NORM",
    KIND_POST_ATTENTION_NORM: "POST_ATTENTION_NORM",
    KIND_PRE_FFN_NORM: "PRE_FFN_NORM",
    KIND_POST_FFN_NORM: "POST_FFN_NORM",
    KIND_QGKV: "QGKV",
    KIND_ATTN_OUTPUT: "ATTN_OUTPUT",
    KIND_ATTN_GATE_UP: "ATTN_GATE_UP",
    KIND_MLP_DOWN: "MLP_DOWN",
}

BF16_BYTES = 2
READ_CHUNK = 512 * 1024


def local_query_head_count(tp_degree: int) -> int:
    return ATTN_QUERY_HEADS // tp_degree


def kv_head_of_rank(tp_degree: int, tp_rank: int) -> int:
    return tp_rank * ATTN_KV_HEADS // tp_degree


class Record:
    __slots__ = ("kind", "layer", "weight_format", "rows", "columns", "payload_bytes", "scale_bytes", "names", "plan")

    def __init__(self, kind: int, layer: int, rows: int, columns: int, names: list, plan: dict) -> None:
        self.kind = kind
        self.layer = layer
        self.weight_format = WEIGHT_BF16
        self.rows = rows
        self.columns = columns
        self.payload_bytes = rows * columns * BF16_BYTES
        self.scale_bytes = 0
        self.names = names
        self.plan = plan

    def label(self) -> str:
        return f"{KIND_NAMES[self.kind]}:layer={self.layer:#x}"


def qgkv_row_spans(tp_degree: int, tp_rank: int) -> list:
    heads = local_query_head_count(tp_degree)
    kv_base = kv_head_of_rank(tp_degree, tp_rank)
    spans = []
    for local in range(heads):
        head = tp_rank * heads + local
        spans.append(("q", head * ATTN_HEAD_DIM, ATTN_HEAD_DIM))
        spans.append(("gate", head * ATTN_HEAD_DIM, ATTN_HEAD_DIM))
    spans.append(("k", kv_base * ATTN_HEAD_DIM, ATTN_HEAD_DIM))
    spans.append(("v", kv_base * ATTN_HEAD_DIM, ATTN_HEAD_DIM))
    return spans


def build_records(tp_degree: int, tp_rank: int) -> list:
    if tp_degree < 1 or tp_rank >= tp_degree:
        raise PackFailure(f"invalid tp rank {tp_rank}/{tp_degree}")
    if ATTN_QUERY_HEADS % tp_degree != 0 or INTERMEDIATE % tp_degree != 0 or VOCAB % tp_degree != 0:
        raise PackFailure(f"geometry does not shard by tp degree {tp_degree}")
    heads = local_query_head_count(tp_degree)
    local_intermediate = INTERMEDIATE // tp_degree
    local_vocab = VOCAB // tp_degree
    records = [
        Record(KIND_EMBEDDING, GLOBAL_LAYER, local_vocab, HIDDEN, ["model.language_model.embed_tokens.weight"],
               {"row_slice": (tp_rank * local_vocab, local_vocab)}),
        Record(KIND_FINAL_NORM, GLOBAL_LAYER, 1, HIDDEN, ["model.language_model.norm.weight"], {"whole": True}),
        Record(KIND_LM_HEAD, GLOBAL_LAYER, local_vocab, HIDDEN, ["lm_head.weight"],
               {"row_slice": (tp_rank * local_vocab, local_vocab)}),
    ]
    norm_kinds = ((KIND_INPUT_NORM, "input_layernorm.weight"),
                  (KIND_POST_ATTENTION_NORM, "post_attention_layernorm.weight"),
                  (KIND_PRE_FFN_NORM, "pre_feedforward_layernorm.weight"),
                  (KIND_POST_FFN_NORM, "post_feedforward_layernorm.weight"))
    qgkv_rows = heads * 2 * ATTN_HEAD_DIM + 2 * ATTN_HEAD_DIM
    for layer in range(LAYER_COUNT):
        prefix = f"{LAYER_PREFIX}{layer}."
        for kind, suffix in norm_kinds:
            records.append(Record(kind, layer, 1, HIDDEN, [prefix + suffix], {"whole": True}))
        records.append(Record(KIND_QGKV, layer, qgkv_rows, HIDDEN,
                              [prefix + f"self_attn.{proj}.weight" for proj in ("q_proj", "gate_proj", "k_proj",
                                                                                "v_proj")],
                              {"qgkv": qgkv_row_spans(tp_degree, tp_rank)}))
        records.append(Record(KIND_ATTN_OUTPUT, layer, HIDDEN, heads * ATTN_HEAD_DIM,
                              [prefix + "self_attn.o_proj.weight"],
                              {"columns": (tp_rank * heads * ATTN_HEAD_DIM, heads * ATTN_HEAD_DIM)}))
        gate_start = tp_rank * local_intermediate
        records.append(Record(KIND_ATTN_GATE_UP, layer, 2 * local_intermediate, HIDDEN,
                              [prefix + "mlp.gate_proj.weight", prefix + "mlp.up_proj.weight"],
                              {"gate_up": (gate_start, local_intermediate)}))
        records.append(Record(KIND_MLP_DOWN, layer, HIDDEN, local_intermediate, [prefix + "mlp.down_proj.weight"],
                              {"columns": (gate_start, local_intermediate)}))
    return records


def build_directory(records: list) -> tuple:
    start = align_up(HEADER_BYTES + ENTRY_BYTES * len(records), PAYLOAD_ALIGNMENT)
    offsets, end = region_offsets(((record.payload_bytes, 0) for record in records), start, align_payloads=False)
    return [(record, payload_offset, 0) for record, (payload_offset, _) in zip(records, offsets)], end


def audit_two_pass(records: list) -> dict:
    first, first_bytes = build_directory(records)
    second, second_bytes = build_directory(records)
    if first_bytes != second_bytes or len(first) != len(second):
        raise PackFailure("two-pass placement proof diverged")
    for (record_a, offset_a, _), (_, offset_b, _) in zip(first, second):
        if offset_a != offset_b:
            raise PackFailure(f"placement drift on kind {record_a.kind}")
    return {"passes": 2, "file_bytes": second_bytes, "already_placed": [record.label() for record, _, _ in second]}


def make_header(tensor_count: int, file_bytes: int) -> bytes:
    return HEADER_STRUCT.pack(MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, tensor_count, HIDDEN, LAYER_COUNT, 0,
                              LAYER_COUNT, ATTENTION_PERIOD, FULL_PHASE, 0, 0, 0, 0, 0, ATTN_QUERY_HEADS, ATTN_KV_HEADS,
                              ATTN_HEAD_DIM, ATTN_HEAD_DIM, 0, 0, INTERMEDIATE, VOCAB, 0, 0, HEADER_BYTES, file_bytes)


def rows_of(source: SafetensorsSource, name: str, first_row: int, row_count: int, row_bytes: int) -> bytes:
    _, _, data_offset = source.resolve(name)
    fd = os.open(source.root / source.weight_map[name], os.O_RDONLY)
    try:
        return b"".join(window_chunks(lambda offset, length: os.pread(fd, length, offset), data_offset, row_bytes,
                                      first_row, row_count, 0, row_bytes, READ_CHUNK, f"on {name}"))
    finally:
        os.close(fd)


def read_matrix(source: SafetensorsSource, name: str):
    shape = source.resolve(name)[1]["shape"]
    rows = shape[0] if len(shape) > 1 else 1
    matrix = np.frombuffer(rows_of(source, name, 0, rows, int(np.prod(shape)) // rows * BF16_BYTES), dtype="<u2")
    return matrix.reshape(shape[0], -1) if len(shape) > 1 else matrix


def record_payload(source: SafetensorsSource, record: Record) -> bytes:
    row_bytes = record.columns * BF16_BYTES
    plan = record.plan
    if "whole" in plan:
        return rows_of(source, record.names[0], 0, record.rows, row_bytes)
    if "row_slice" in plan:
        return rows_of(source, record.names[0], *plan["row_slice"], row_bytes)
    if "qgkv" in plan:
        planes = dict(zip(("q", "gate", "k", "v"), (read_matrix(source, name) for name in record.names)))
        blocks = []
        for plane_name, first_row, row_count in plan["qgkv"]:
            if planes[plane_name].shape[0] < first_row + row_count:
                raise PackFailure(f"qgkv span out of range ({plane_name})")
            blocks.append(planes[plane_name][first_row:first_row + row_count, :])
        return np.ascontiguousarray(np.concatenate(blocks, axis=0)).tobytes()
    if "gate_up" in plan:
        start, count = plan["gate_up"]
        gate, up = (read_matrix(source, name) for name in record.names)
        if gate.shape[0] < start + count or up.shape[0] < start + count:
            raise PackFailure("gate/up rows out of range")
        return np.ascontiguousarray(np.concatenate((gate[start:start + count, :], up[start:start + count, :]),
                                                   axis=0)).tobytes()
    start, count = plan["columns"]
    full = read_matrix(source, record.names[0])
    if full.shape[1] < start + count:
        raise PackFailure(f"{record.names[0]}: columns {full.shape[1]} < {start + count}")
    return np.ascontiguousarray(full[:, start:start + count]).tobytes()


def write_records(source: SafetensorsSource, entries: list, output: Path) -> None:
    fd = os.open(output, os.O_WRONLY)
    try:
        for record, payload_offset, _ in entries:
            payload = record_payload(source, record)
            if len(payload) != record.payload_bytes:
                raise PackFailure(f"{record.label()} payload size mismatch")
            os.pwrite(fd, payload, payload_offset)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="warm checkpoint root (safetensors + config)")
    parser.add_argument("--output", type=Path, required=True, help="output pack path for this rank")
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="inventory + two-pass proof, no source reads")
    arguments = parser.parse_args()
    if arguments.tp_degree < 1 or arguments.tp_rank >= arguments.tp_degree:
        raise PackFailure(f"invalid tp rank {arguments.tp_rank}/{arguments.tp_degree}")
    records = build_records(arguments.tp_degree, arguments.tp_rank)
    proof = audit_two_pass(records)
    entries, file_bytes = build_directory(records)
    kind_counts = {}
    for record, _, _ in entries:
        kind_counts[KIND_NAMES[record.kind]] = kind_counts.get(KIND_NAMES[record.kind], 0) + 1
    receipt = {
        "family": "muse_glimmer",
        "magic": f"0x{MAGIC:08X}",
        "format_version": FORMAT_VERSION,
        "tp_degree": arguments.tp_degree,
        "tp_rank": arguments.tp_rank,
        "kv_head_of_rank": kv_head_of_rank(arguments.tp_degree, arguments.tp_rank),
        "tensor_count": len(records),
        "file_bytes": file_bytes,
        "kinds": kind_counts,
        "census_expectations": json.loads(DEFAULT_PATTERNS.read_text())["census"],
        "two_pass_placement_proof": proof,
    }
    if not arguments.dry_run:
        if arguments.source is None:
            raise PackFailure("--source is required without --dry-run")
        source = SafetensorsSource(arguments.source)
        source.check_config({"hidden_size": HIDDEN, "num_hidden_layers": LAYER_COUNT, "vocab_size": VOCAB,
                             "num_attention_heads": ATTN_QUERY_HEADS, "num_key_value_heads": ATTN_KV_HEADS,
                             "head_dim": ATTN_HEAD_DIM, "intermediate_size": INTERMEDIATE}, section="text_config")
        with open(arguments.output, "wb") as out:
            out.write(make_header(len(records), file_bytes))
            for record, payload_offset, _ in entries:
                out.write(ENTRY_STRUCT.pack(record.kind, record.layer, record.weight_format, record.rows,
                                            record.columns, 0, payload_offset, record.payload_bytes, 0,
                                            record.scale_bytes))
        write_records(source, entries, arguments.output)
        receipt["pack_sha256"] = sha256_file(arguments.output)
        receipt["index_sha256"] = source.index_sha256
        receipt["config_sha256"] = source.config_sha256
    else:
        receipt["dry_run"] = True
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(receipt, arguments.output)
    print(json.dumps({key: receipt[key] for key in receipt if key != "two_pass_placement_proof"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
