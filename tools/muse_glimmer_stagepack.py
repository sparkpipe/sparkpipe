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
import struct
import sys
from pathlib import Path

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
DEFAULT_PATTERNS = ROOT / "model-families" / "muse_glimmer" / "tensor_patterns.json"

MAGIC = 0x47534D55  # 'GSMU' little endian
FORMAT_VERSION = 1
HEADER_BYTES = 120
ENTRY_BYTES = 56
GLOBAL_LAYER = 0xFFFFFFFF
PAYLOAD_ALIGNMENT = 256
WEIGHT_BF16 = 0

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

HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES

BF16_BYTES = 2
READ_CHUNK = 512 * 1024


def local_query_head_count(tp_degree: int) -> int:
    return ATTN_QUERY_HEADS // tp_degree


def kv_head_of_rank(tp_degree: int, tp_rank: int) -> int:
    return tp_rank * ATTN_KV_HEADS // tp_degree


class Record:
    __slots__ = ("kind", "layer", "weight_format", "rows", "columns",
                 "payload_bytes", "scale_bytes", "names", "plan")

    def __init__(self, kind: int, layer: int, rows: int, columns: int,
                 names: list, plan: dict) -> None:
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
    """Stagepack qgkv rows as (source_name, first_row, row_count) spans in
    split order: per local head q then the output gate, then this rank's
    replicated kv head's k and v."""
    heads = local_query_head_count(tp_degree)
    kv_base = kv_head_of_rank(tp_degree, tp_rank)
    name_q = "model.language_model.embed_tokens.weight"  # placeholder no
    spans = []
    del name_q
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
    if ATTN_QUERY_HEADS % tp_degree != 0 or INTERMEDIATE % tp_degree != 0 \
            or VOCAB % tp_degree != 0:
        raise PackFailure(f"geometry does not shard by tp degree {tp_degree}")
    records = []
    heads = local_query_head_count(tp_degree)
    local_intermediate = INTERMEDIATE // tp_degree
    local_vocab = VOCAB // tp_degree
    records.append(Record(KIND_EMBEDDING, GLOBAL_LAYER, local_vocab, HIDDEN,
                          ["model.language_model.embed_tokens.weight"],
                          {"row_slice": (tp_rank * local_vocab, local_vocab)}))
    records.append(Record(KIND_FINAL_NORM, GLOBAL_LAYER, 1, HIDDEN,
                          ["model.language_model.norm.weight"],
                          {"whole": True}))
    records.append(Record(KIND_LM_HEAD, GLOBAL_LAYER, local_vocab, HIDDEN,
                          ["lm_head.weight"],
                          {"row_slice": (tp_rank * local_vocab, local_vocab)}))
    norm_kinds = (
        (KIND_INPUT_NORM, "input_layernorm.weight"),
        (KIND_POST_ATTENTION_NORM, "post_attention_layernorm.weight"),
        (KIND_PRE_FFN_NORM, "pre_feedforward_layernorm.weight"),
        (KIND_POST_FFN_NORM, "post_feedforward_layernorm.weight"),
    )
    qgkv_rows = heads * 2 * ATTN_HEAD_DIM + 2 * ATTN_HEAD_DIM
    for layer in range(LAYER_COUNT):
        for kind, suffix in norm_kinds:
            records.append(Record(kind, layer, 1, HIDDEN,
                                  [f"{LAYER_PREFIX}{layer}.{suffix}"],
                                  {"whole": True}))
        records.append(Record(
            KIND_QGKV, layer, qgkv_rows, HIDDEN,
            [f"{LAYER_PREFIX}{layer}.self_attn.q_proj.weight",
             f"{LAYER_PREFIX}{layer}.gate_proj.weight",
             f"{LAYER_PREFIX}{layer}.self_attn.k_proj.weight",
             f"{LAYER_PREFIX}{layer}.self_attn.v_proj.weight"],
            {"qgkv": qgkv_row_spans(tp_degree, tp_rank)}))
        records.append(Record(
            KIND_ATTN_OUTPUT, layer, HIDDEN, heads * ATTN_HEAD_DIM,
            [f"{LAYER_PREFIX}{layer}.self_attn.o_proj.weight"],
            {"columns": (tp_rank * heads * ATTN_HEAD_DIM,
                         heads * ATTN_HEAD_DIM)}))
        gate_start = tp_rank * local_intermediate
        records.append(Record(
            KIND_ATTN_GATE_UP, layer, 2 * local_intermediate, HIDDEN,
            [f"{LAYER_PREFIX}{layer}.mlp.gate_proj.weight",
             f"{LAYER_PREFIX}{layer}.mlp.up_proj.weight"],
            {"gate_up": (gate_start, local_intermediate)}))
        records.append(Record(
            KIND_MLP_DOWN, layer, HIDDEN, local_intermediate,
            [f"{LAYER_PREFIX}{layer}.mlp.down_proj.weight"],
            {"columns": (gate_start, local_intermediate)}))
    return records


def build_directory(records: list) -> tuple:
    cursor = align_up(HEADER_BYTES + ENTRY_BYTES * len(records),
                      PAYLOAD_ALIGNMENT)
    entries = []
    for record in records:
        entries.append((record, cursor, 0))
        cursor += record.payload_bytes
    return entries, cursor


def audit_two_pass(records: list) -> dict:
    """Placement two-pass proof: rebuilding the directory must reproduce the
    first pass byte for byte."""
    first, first_bytes = build_directory(records)
    second, second_bytes = build_directory(records)
    if first_bytes != second_bytes or len(first) != len(second):
        raise PackFailure("two-pass placement proof diverged")
    for (record_a, offset_a, _), (record_b, offset_b, _) in zip(first, second):
        if offset_a != offset_b:
            raise PackFailure(f"placement drift on kind {record_a.kind}")
    return {"passes": 2, "file_bytes": second_bytes,
            "already_placed": [record.label() for record, _, _ in second]}


def make_header(tp_degree: int, tensor_count: int, file_bytes: int) -> bytes:
    del tp_degree
    return HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES,
        tensor_count, HIDDEN, LAYER_COUNT, 0, LAYER_COUNT,
        ATTENTION_PERIOD, FULL_PHASE, 0, 0, 0, 0, 0,
        ATTN_QUERY_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM, ATTN_HEAD_DIM,
        0, 0, INTERMEDIATE, VOCAB, 0, 0,
        HEADER_BYTES + ENTRY_BYTES * tensor_count, file_bytes)


def read_matrix(source: SafetensorsSource, name: str):
    """Whole bf16 matrix as a numpy uint16 view (pread slices, no torch)."""
    import numpy as np
    shard, meta, data_offset = source.resolve(name)
    shape = meta["shape"]
    elements = 1
    for extent in shape:
        elements *= extent
    raw = bytearray(elements * BF16_BYTES)
    fd = os.open(source.root / source.weight_map[name], os.O_RDONLY)
    try:
        view = memoryview(raw)
        position, left = data_offset, len(raw)
        while left > 0:
            step = min(left, READ_CHUNK)
            chunk = os.pread(fd, step, position)
            if len(chunk) != step:
                raise PackFailure(f"short read on {name}")
            view[len(raw) - left:len(raw) - left + step] = chunk
            position += step
            left -= step
    finally:
        os.close(fd)
    matrix = np.frombuffer(raw, dtype="<u2")
    return matrix.reshape(shape[0], -1) if len(shape) > 1 else matrix


def stream_row_span(fd_out: int, source: SafetensorsSource, name: str,
                    first_row: int, row_count: int, row_bytes: int,
                    payload_offset: int) -> None:
    """Contiguous bf16 rows: pread spans straight into the pack file."""
    shard, meta, data_offset = source.resolve(name)
    fd_in = os.open(source.root / source.weight_map[name], os.O_RDONLY)
    try:
        position = data_offset + first_row * row_bytes
        left = row_count * row_bytes
        written = 0
        while left > 0:
            step = min(left, READ_CHUNK)
            if step > row_bytes:
                step -= step % row_bytes
            chunk = os.pread(fd_in, step, position)
            if len(chunk) != step:
                raise PackFailure(f"short read on {name}")
            if os.lseek(fd_out, payload_offset + written, os.SEEK_SET) < 0:
                raise PackFailure("seek failed")
            os.write(fd_out, chunk)
            written += step
            position += step
            left -= step
    finally:
        os.close(fd_in)


def write_records(source: SafetensorsSource, entries: list, output: Path) -> None:
    fd = os.open(output, os.O_WRONLY)
    try:
        for record, payload_offset, _ in entries:
            row_bytes = record.columns * BF16_BYTES
            if "whole" in record.plan:
                stream_row_span(fd, source, record.names[0], 0,
                                record.rows, row_bytes, payload_offset)
            elif "row_slice" in record.plan:
                first, count = record.plan["row_slice"]
                stream_row_span(fd, source, record.names[0], first, count,
                                row_bytes, payload_offset)
            elif "qgkv" in record.plan:
                write_qgkv(fd, source, record, payload_offset)
            elif "gate_up" in record.plan:
                write_gate_up(fd, source, record, payload_offset)
            else:
                write_columns(fd, source, record, payload_offset)
    finally:
        os.close(fd)


def write_qgkv(fd: int, source: SafetensorsSource, record: Record,
               payload_offset: int) -> None:
    import numpy as np
    name_q, name_gate, name_k, name_v = record.names
    planes = {"q": read_matrix(source, name_q),
              "gate": read_matrix(source, name_gate),
              "k": read_matrix(source, name_k),
              "v": read_matrix(source, name_v)}
    blocks = []
    for plane_name, first_row, row_count in record.plan["qgkv"]:
        plane = planes[plane_name]
        if plane.shape[0] < first_row + row_count:
            raise PackFailure(f"qgkv span out of range ({plane_name})")
        blocks.append(plane[first_row:first_row + row_count, :])
    payload = np.ascontiguousarray(np.concatenate(blocks, axis=0)).tobytes()
    if len(payload) != record.payload_bytes:
        raise PackFailure("qgkv payload size mismatch")
    os.lseek(fd, payload_offset, os.SEEK_SET)
    os.write(fd, payload)


def write_gate_up(fd: int, source: SafetensorsSource, record: Record,
                  payload_offset: int) -> None:
    import numpy as np
    gate_start, count = record.plan["gate_up"]
    gate = read_matrix(source, record.names[0])
    up = read_matrix(source, record.names[1])
    if gate.shape[0] < gate_start + count or up.shape[0] < gate_start + count:
        raise PackFailure("gate/up rows out of range")
    gathered = np.concatenate((
        np.ascontiguousarray(gate[gate_start:gate_start + count, :]),
        np.ascontiguousarray(up[gate_start:gate_start + count, :])), axis=0)
    payload = np.ascontiguousarray(gathered).tobytes()
    if len(payload) != record.payload_bytes:
        raise PackFailure("gate_up payload size mismatch")
    os.lseek(fd, payload_offset, os.SEEK_SET)
    os.write(fd, payload)


def write_columns(fd: int, source: SafetensorsSource, record: Record,
                  payload_offset: int) -> None:
    import numpy as np
    start, count = record.plan["columns"]
    full = read_matrix(source, record.names[0])
    if full.shape[1] < start + count:
        raise PackFailure(f"{record.names[0]}: columns {full.shape[1]} < {start + count}")
    payload = np.ascontiguousarray(full[:, start:start + count]).tobytes()
    if len(payload) != record.payload_bytes:
        raise PackFailure("column payload size mismatch")
    os.lseek(fd, payload_offset, os.SEEK_SET)
    os.write(fd, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        help="warm checkpoint root (safetensors + config)")
    parser.add_argument("--output", type=Path, required=True,
                        help="output pack path for this rank")
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="inventory + two-pass proof, no source reads")
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
        source.check_config({
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYER_COUNT,
            "vocab_size": VOCAB,
            "num_attention_heads": ATTN_QUERY_HEADS,
            "num_key_value_heads": ATTN_KV_HEADS,
            "head_dim": ATTN_HEAD_DIM,
            "intermediate_size": INTERMEDIATE,
        }, section="text_config")
        with open(arguments.output, "wb") as out:
            out.write(make_header(arguments.tp_degree, len(records), file_bytes))
            for record, payload_offset, _ in entries:
                # Field order is the frozen SparkMuseGlimmerStagePackEntry
                # wire layout (<6I4Q>): kind, layer, weight_format, rows,
                # columns, scale_group_size, payload_offset, payload_bytes,
                # scale_offset, scale_bytes.
                out.write(ENTRY_STRUCT.pack(record.kind, record.layer,
                                            record.weight_format, record.rows,
                                            record.columns, 0,
                                            payload_offset, record.payload_bytes,
                                            0, record.scale_bytes))
        write_records(source, entries, arguments.output)
        receipt["pack_sha256"] = sha256_file(arguments.output)
        receipt["index_sha256"] = source.index_sha256
        receipt["config_sha256"] = source.config_sha256
    else:
        receipt["dry_run"] = True
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(receipt, arguments.output)
    print(json.dumps({key: receipt[key] for key in receipt
                      if key != "two_pass_placement_proof"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
