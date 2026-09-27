#!/usr/bin/env python3
"""Convert the google/gemma-4 warm checkpoints into gemma4 stage packs
(lane gemma4, AC8).

Setup-time code, never the serving path. Mirrors tools/qwen38_stagepack.py
and tools/spark_pack_common.py for the safetensors streaming, re-parameterized
for the gemma4 geometry frozen in model_contracts/gemma4_*_authoritative.json.

Per-model pack shapes (per-rank, TP-sharded, per the frozen rulings):

  * 31B dense, TP16xPP1: sliding kv 16/16 exact per rank; full kv 4 global
    heads replicated x4, rank r reads global head r/4; per-rank KV_HEADS 1
    both kinds; vocab shard 16384 rows/rank; per-rank census 663 tensors/60L.
  * 26B-A4B MoE, TP4xPP4 stage lists {8,8,7,7}: sliding kv 8/4 = 2 per rank;
    full kv 2 global heads replicated x2, rank r reads head r/2; experts
    32/rank; vocab shard 65536 rows/rank.

Pack-time transforms, each ruled in the contract pack_folds:

  * sliding k_proj|v_proj row fusion into the SLIDING_KV_FUSED slot.
  * router.scale (per-hidden vector, bf16) folded into router proj columns
    with the H**-0.5 factor: proj[:,i] *= scale[i] * hidden**-0.5.
  * router.per_expert_scale folded into expert down rows
    (down[e] *= per_expert_scale[e]); the PER_EXPERT_SCALE slot still emits
    the f32-converted raw vector for the nvfp4 seam, never re-applied by the
    module.
  * full-layer inv_freq table emitted as a global f32 tensor:
    1e6**(-i/256) for i < 64 then 0 (64 rotated pairs, publisher-derived).
  * layer_scalar asserted exactly 1.0 for every layer, fail closed.

The 26B MoE arm additionally writes a version-2 experts manifest
(48-byte records: layer, expert, kind=tensor_kind*2+plane, offset, bytes,
ck128) covering both expert planes per rank, and the receipt reports the
spine/expert byte split for MEMORY_MODEL.md.

Placement is proven in two passes: pass 1 computes the full directory
(offsets, alignment, sizes) and its sha256; pass 2 streams the payloads at
those offsets and a verify walk re-reads every directory entry and the
header, asserting byte-exact agreement with the pass-1 plan.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
import numpy as np

from spark_pack_common import (  # noqa: E402
    PackFailure,
    SafetensorsSource,
    align_up,
    sha256_bytes,
    sha256_file,
    tp_shard_range,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

MAGIC = 0x50534734
FORMAT_VERSION = 1
HEADER_BYTES = 120
ENTRY_BYTES = 56
GLOBAL_LAYER = 0xFFFFFFFF
PAYLOAD_ALIGNMENT = 256
WEIGHT_BF16 = 0
WEIGHT_F32 = 1

HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES

EXPERT_MANIFEST_MAGIC = 0x58504557
EXPERT_MANIFEST_VERSION = 2
EXPERT_RECORD_STRUCT = struct.Struct("<IIIIQQ4s")

GEOMETRY = {
    "31b": dict(
        model_id="google/gemma-4-31B-it",
        hidden=5376, layers=60, vocab=262144,
        sliding_q_heads=32, sliding_kv_heads=16, sliding_head_dim=256,
        full_q_heads=32, full_kv_heads=4, full_head_dim=512,
        dense_inter=21504, window=1024,
        experts=None, experts_per_token=None, expert_inter=None,
        topology="tp16pp1", stage_lists=[(0, 60)],
    ),
    "26b-a4b": dict(
        model_id="google/gemma-4-26B-A4B-it",
        hidden=2816, layers=30, vocab=262144,
        sliding_q_heads=16, sliding_kv_heads=8, sliding_head_dim=256,
        full_q_heads=16, full_kv_heads=2, full_head_dim=512,
        dense_inter=2112, window=1024,
        experts=128, experts_per_token=8, expert_inter=704,
        topology="tp4pp4", stage_lists=[(0, 8), (8, 16), (16, 23), (23, 30)],
    ),
}

(KIND_EMBEDDING, KIND_FINAL_NORM, KIND_INPUT_NORM, KIND_POST_ATTENTION_NORM,
 KIND_PRE_FF_NORM, KIND_POST_FF_NORM, KIND_SLIDING_QUERY, KIND_SLIDING_KV,
 KIND_SLIDING_OUTPUT, KIND_SLIDING_Q_NORM, KIND_SLIDING_K_NORM,
 KIND_FULL_QUERY, KIND_FULL_KEY, KIND_FULL_OUTPUT, KIND_FULL_Q_NORM,
 KIND_FULL_K_NORM, KIND_MLP_GATE_UP, KIND_MLP_DOWN, KIND_ROPE_TABLE,
 KIND_ROUTER_PROJ, KIND_PER_EXPERT_SCALE, KIND_EXPERT_GATE_UP,
 KIND_EXPERT_DOWN, KIND_POST_FF_1_NORM, KIND_PRE_FF_2_NORM,
 KIND_POST_FF_2_NORM, KIND_LAYER_SCALAR) = range(27)

BASE_NORM_KINDS = (KIND_INPUT_NORM, KIND_POST_ATTENTION_NORM,
                   KIND_PRE_FF_NORM, KIND_POST_FF_NORM)
MOE_NORM_KINDS = (KIND_POST_FF_1_NORM, KIND_PRE_FF_2_NORM,
                  KIND_POST_FF_2_NORM)

MOE_NORM_SOURCES = {
    KIND_POST_FF_1_NORM: "model.language_model.layers.{layer}.post_feedforward_layernorm_1.weight",
    KIND_PRE_FF_2_NORM: "model.language_model.layers.{layer}.pre_feedforward_layernorm_2.weight",
    KIND_POST_FF_2_NORM: "model.language_model.layers.{layer}.post_feedforward_layernorm_2.weight",
}


def bf16_blocks_to_f32(packed: bytes) -> bytes:
    words = np.frombuffer(packed, dtype="<u2").astype("<u4") << 16
    return words.tobytes()


E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                       dtype=np.float32)
E2M1_LUT = np.concatenate([E2M1_VALUES, -E2M1_VALUES]).astype(np.float32)


def e4m3_lut() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint32)
    sign = (codes >> 7) & 1
    exp = (codes >> 3) & 0xF
    man = codes & 0x7
    value = np.where(
        exp == 0,
        man.astype(np.float32) * np.float32(2.0 ** -9),
        (np.float32(1.0) + man.astype(np.float32) / np.float32(8.0))
        * np.power(np.float32(2.0), (exp.astype(np.int32) - 7).astype(np.float32)),
    ).astype(np.float32)
    value[(exp == 15) & (man == 7)] = np.float32("nan")
    return np.where(sign == 1, -value, value).astype(np.float32)


E4M3_LUT = e4m3_lut()


def read_nvfp4_matrix_bf16(source: SafetensorsSource, stem: str, rows: int,
                           columns: int, row_start: int = 0,
                           row_count: int | None = None,
                           column_start: int = 0,
                           column_count: int | None = None) -> bytes:
    """Decode one modelopt nvfp4 projection window into bf16 bytes.

    W = e2m1(packed U8 [rows, cols/2], low nibble first) x
        e4m3(weight_scale [rows, cols/16]) x weight_scale_2 (F32 scalar),
        one round-to-nearest-even bf16 rounding - the modelopt nvfp4
        release's defined weight values (pinned elementwise against the
        bf16 release twin). input_scale is the activation-side scale and
        never enters the weight decode. Only the requested row/column
        window is materialized; column windows must be 16-aligned so the
        per-16 scale plane slices without regrouping."""
    if row_count is None:
        row_count = rows - row_start
    if column_count is None:
        column_count = columns - column_start
    if column_start % 16 or column_count % 16:
        raise PackFailure(f"{stem}: nvfp4 column window "
                          f"[{column_start}, {column_start + column_count}) "
                          f"is not 16-aligned")
    shard, meta, data_start = source.resolve(stem + ".weight")
    if meta["dtype"] != "U8" or meta["shape"] != [rows, columns // 2]:
        raise PackFailure(f"{stem}.weight: {meta['dtype']} {meta['shape']}, "
                          f"expected U8 [{rows}, {columns // 2}]")
    s_shard, s_meta, s_start = source.resolve(stem + ".weight_scale")
    if s_meta["dtype"] != "F8_E4M3" or s_meta["shape"] != [rows, columns // 16]:
        raise PackFailure(f"{stem}.weight_scale: {s_meta['dtype']} {s_meta['shape']}, "
                          f"expected F8_E4M3 [{rows}, {columns // 16}]")
    g_shard, g_meta, g_start = source.resolve(stem + ".weight_scale_2")
    if g_meta["dtype"] != "F32" or tuple(g_meta["shape"]) not in ((), (1,)):
        raise PackFailure(f"{stem}.weight_scale_2: {g_meta['dtype']} "
                          f"{g_meta['shape']}, expected F32 scalar")
    with (source.root / g_shard).open("rb") as file:
        file.seek(g_start)
        global_scale = struct.unpack("<f", file.read(4))[0]
    group = 16
    row_bytes = columns // 2
    byte_start = column_start // 2
    byte_count = column_count // 2
    group_start = column_start // group
    group_count = column_count // group
    out = bytearray()
    with (source.root / shard).open("rb") as file, (source.root / s_shard).open("rb") as sfile:
        for row in range(row_start, row_start + row_count):
            file.seek(data_start + row * row_bytes + byte_start)
            codes = np.frombuffer(file.read(byte_count), dtype=np.uint8)
            sfile.seek(s_start + row * (columns // group)
                       + group_start)
            plane = np.frombuffer(sfile.read(group_count), dtype=np.uint8)
            values = np.empty(column_count, dtype=np.float32)
            values[0::2] = E2M1_LUT[codes & 0xF]
            values[1::2] = E2M1_LUT[codes >> 4]
            expanded = np.repeat(E4M3_LUT[plane], group)[:column_count]
            u16 = bf16_round_array(values * expanded * global_scale)
            out += u16.tobytes()
    if len(out) != row_count * column_count * 2:
        raise PackFailure(f"{stem}: nvfp4 decode produced {len(out)} bytes, "
                          f"expected {row_count * column_count * 2}")
    return bytes(out)


def read_projection_bf16(source: SafetensorsSource, stem: str, rows: int,
                         columns: int, row_start: int = 0,
                         row_count: int | None = None,
                         column_start: int = 0,
                         column_count: int | None = None) -> bytes:
    """A dense projection from either release: native BF16 verbatim, or
    the nvfp4 release's packed payload decoded to its defined bf16
    values. This is the only checkpoint-driven branch in the packer."""
    _, meta, _ = source.resolve(stem + ".weight")
    if meta["dtype"] == "U8":
        return read_nvfp4_matrix_bf16(source, stem, rows, columns,
                                      row_start, row_count,
                                      column_start, column_count)
    return read_matrix(source, stem + ".weight", rows, columns,
                       row_start, row_count, column_start, column_count)


def read_matrix(source: SafetensorsSource, name: str, rows: int, columns: int,
                row_start: int = 0, row_count: int | None = None,
                column_start: int = 0, column_count: int | None = None,
                expected_shape: list | None = None) -> bytes:
    """Stream a bf16 row/column slice of one safetensors tensor. A 3-D (or
    higher) checkpoint tensor is addressed flattened: leading dimensions fold
    into rows, the last dimension is columns."""
    if column_count is None:
        column_count = columns - column_start
    if row_count is None:
        row_count = rows - row_start
    shard, meta, data_start = source.resolve(name)
    dtype, shape = meta["dtype"], meta["shape"]
    if dtype != "BF16":
        raise PackFailure(f"{name}: dtype {dtype}, expected BF16 (never quantize)")
    if expected_shape is None:
        expected_shape = [rows, columns]
    if shape != expected_shape:
        raise PackFailure(f"{name}: shape {shape}, expected {expected_shape}")
    shape_product = 1
    for dim in expected_shape:
        shape_product *= dim
    if rows * columns != shape_product:
        raise PackFailure(f"{name}: caller rows*columns {rows}x{columns} does not "
                          f"match expected shape product {expected_shape}")
    row_bytes = columns * 2
    slice_bytes = row_count * column_count * 2
    with (source.root / shard).open("rb") as file:
        pieces = []
        for row in range(row_count):
            file.seek(data_start + (row_start + row) * row_bytes + column_start * 2)
            pieces.append(file.read(column_count * 2))
    if b"".join(pieces).__len__() != slice_bytes:
        raise PackFailure(f"{name}: short read")
    return b"".join(pieces)


def read_vector(source: SafetensorsSource, name: str, count: int) -> bytes:
    shard, meta, data_start = source.resolve(name)
    if meta["dtype"] != "BF16" or meta["shape"] != [count]:
        raise PackFailure(f"{name}: dtype/shape {meta}, expected BF16 [{count}]")
    with (source.root / shard).open("rb") as file:
        file.seek(data_start)
        payload = file.read(count * 2)
    if len(payload) != count * 2:
        raise PackFailure(f"{name}: short read")
    return payload


def read_layer_scalar(source: SafetensorsSource, layer: int) -> float:
    payload = read_vector(source, f"model.language_model.layers.{layer}.layer_scalar", 1)
    bits = payload[0] | (payload[1] << 8)
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def kv_heads_per_rank(global_kv_heads: int, tp_degree: int) -> tuple[int, int]:
    if global_kv_heads >= tp_degree:
        return global_kv_heads // tp_degree, 1
    return 1, tp_degree // global_kv_heads


def layer_is_full(geometry: dict, layer: int) -> bool:
    return (layer % 6) == 5


def build_inventory(geometry: dict, tp_degree: int, tp_rank: int,
                    first_layer: int, layer_count: int) -> list[dict]:
    """Per-rank tensor plan: kind, layer, rows, columns, weight_format."""
    hidden = geometry["hidden"]
    plan: list[dict] = []

    def add(kind: int, layer: int, rows: int, columns: int, fmt: int = WEIGHT_BF16) -> None:
        plan.append(dict(kind=kind, layer=layer, rows=rows, columns=columns,
                         weight_format=fmt))

    add(KIND_ROPE_TABLE, GLOBAL_LAYER, 1, 256, WEIGHT_F32)
    if first_layer == 0:
        add(KIND_EMBEDDING, GLOBAL_LAYER, geometry["vocab"] // tp_degree, hidden)
    for layer in range(first_layer, first_layer + layer_count):
        full = layer_is_full(geometry, layer)
        add(KIND_INPUT_NORM, layer, 1, hidden)
        add(KIND_POST_ATTENTION_NORM, layer, 1, hidden)
        add(KIND_PRE_FF_NORM, layer, 1, hidden)
        add(KIND_POST_FF_NORM, layer, 1, hidden)
        add(KIND_LAYER_SCALAR, layer, 1, 1)
        if geometry["experts"] is not None:
            for kind in MOE_NORM_KINDS:
                add(kind, layer, 1, hidden)
        if full:
            q_heads = geometry["full_q_heads"] // tp_degree
            kv_heads, _ = kv_heads_per_rank(geometry["full_kv_heads"], tp_degree)
            add(KIND_FULL_QUERY, layer, q_heads * geometry["full_head_dim"], hidden)
            add(KIND_FULL_KEY, layer, kv_heads * geometry["full_head_dim"], hidden)
            add(KIND_FULL_OUTPUT, layer, hidden,
                (geometry["full_q_heads"] // tp_degree) * geometry["full_head_dim"])
            add(KIND_FULL_Q_NORM, layer, 1, geometry["full_head_dim"])
            add(KIND_FULL_K_NORM, layer, 1, geometry["full_head_dim"])
        else:
            q_heads = geometry["sliding_q_heads"] // tp_degree
            kv_heads, _ = kv_heads_per_rank(geometry["sliding_kv_heads"], tp_degree)
            add(KIND_SLIDING_QUERY, layer, q_heads * geometry["sliding_head_dim"], hidden)
            add(KIND_SLIDING_KV, layer,
                2 * kv_heads * geometry["sliding_head_dim"], hidden)
            add(KIND_SLIDING_OUTPUT, layer, hidden,
                (geometry["sliding_q_heads"] // tp_degree) * geometry["sliding_head_dim"])
            add(KIND_SLIDING_Q_NORM, layer, 1, geometry["sliding_head_dim"])
            add(KIND_SLIDING_K_NORM, layer, 1, geometry["sliding_head_dim"])
        gate_rows = (2 * geometry["dense_inter"]) // tp_degree
        add(KIND_MLP_GATE_UP, layer, gate_rows, hidden)
        add(KIND_MLP_DOWN, layer, hidden, geometry["dense_inter"] // tp_degree)
        if geometry["experts"] is not None:
            experts_per_rank = geometry["experts"] // tp_degree
            add(KIND_ROUTER_PROJ, layer, geometry["experts"], hidden)
            add(KIND_PER_EXPERT_SCALE, layer, 1, geometry["experts"], WEIGHT_F32)
            add(KIND_EXPERT_GATE_UP, layer,
                experts_per_rank * 2 * geometry["expert_inter"], hidden)
            add(KIND_EXPERT_DOWN, layer,
                experts_per_rank * hidden, geometry["expert_inter"])
    if first_layer + layer_count == geometry["layers"]:
        add(KIND_FINAL_NORM, GLOBAL_LAYER, 1, hidden)
    return plan


def expected_tensor_count(geometry: dict, first_layer: int, layer_count: int) -> int:
    moe = geometry["experts"] is not None
    per_layer = 14 if moe else 7
    tensors = layer_count * (per_layer + 5) + 1
    if first_layer == 0:
        tensors += 1
    if first_layer + layer_count == geometry["layers"]:
        tensors += 1
    return tensors


def place(plan: list[dict]) -> tuple[list[dict], int, int]:
    """Pass 1: directory layout with 256-byte payload alignment."""
    directory_offset = HEADER_BYTES
    directory_bytes = len(plan) * ENTRY_BYTES
    cursor = align_up(directory_offset + directory_bytes, PAYLOAD_ALIGNMENT)
    total_payload = 0
    for entry in plan:
        elements = entry["rows"] * entry["columns"]
        if entry["weight_format"] == WEIGHT_F32:
            entry["payload_bytes"] = elements * 4
        else:
            entry["payload_bytes"] = elements * 2
        entry["payload_offset"] = cursor
        entry["scale_offset"] = 0
        entry["scale_bytes"] = 0
        cursor = align_up(cursor + entry["payload_bytes"], PAYLOAD_ALIGNMENT)
        total_payload += entry["payload_bytes"]
    file_bytes = align_up(cursor, PAYLOAD_ALIGNMENT)
    return plan, file_bytes, total_payload


def header_fields(geometry: dict, plan: list[dict], first_layer: int,
                  layer_count: int, file_bytes: int, directory_offset: int) -> list:
    moe = geometry["experts"] is not None
    return [
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(plan),
        geometry["hidden"], layer_count, first_layer, geometry["layers"], 6, 5,
        geometry["sliding_kv_heads"], geometry["dense_inter"],
        geometry["sliding_head_dim"], geometry["full_head_dim"],
        geometry["window"], geometry["sliding_q_heads"],
        geometry["full_kv_heads"], geometry["sliding_head_dim"],
        geometry["full_head_dim"],
        geometry["experts"] if moe else 0,
        geometry["experts_per_token"] if moe else 0,
        geometry["expert_inter"] if moe else 0,
        geometry["vocab"], 2, 0, directory_offset, file_bytes,
    ]


def payload_for(source: SafetensorsSource, geometry: dict, entry: dict,
                tp_degree: int, tp_rank: int) -> bytes:
    kind, layer = entry["kind"], entry["layer"]
    hidden = geometry["hidden"]
    prefix = "model.language_model.layers"
    if kind == KIND_ROPE_TABLE:
        return struct.pack("<256f", *[
            (1e6 ** (-2.0 * i / 512.0)) if i < 64 else 0.0 for i in range(256)])
    if kind == KIND_EMBEDDING:
        row_start, row_count = tp_shard_range(geometry["vocab"], tp_degree, tp_rank)
        return read_matrix(source, "model.language_model.embed_tokens.weight",
                           geometry["vocab"], hidden, row_start, row_count)
    if kind == KIND_LAYER_SCALAR:
        return read_vector(source, f"{prefix}.{layer}.layer_scalar", 1)
    if kind == KIND_FINAL_NORM:
        return read_vector(source, "model.language_model.norm.weight", hidden)
    if kind in BASE_NORM_KINDS:
        names = {
            KIND_INPUT_NORM: "input_layernorm.weight",
            KIND_POST_ATTENTION_NORM: "post_attention_layernorm.weight",
            KIND_PRE_FF_NORM: "pre_feedforward_layernorm.weight",
            KIND_POST_FF_NORM: "post_feedforward_layernorm.weight",
        }
        return read_vector(source, f"{prefix}.{layer}.{names[kind]}", hidden)
    if kind in MOE_NORM_KINDS:
        return read_vector(source, MOE_NORM_SOURCES[kind].format(layer=layer), hidden)
    if kind in (KIND_SLIDING_Q_NORM, KIND_SLIDING_K_NORM,
                KIND_FULL_Q_NORM, KIND_FULL_K_NORM):
        head_dim = geometry["sliding_head_dim"] if kind in (KIND_SLIDING_Q_NORM, KIND_SLIDING_K_NORM) \
            else geometry["full_head_dim"]
        part = "q_norm.weight" if kind in (KIND_SLIDING_Q_NORM, KIND_FULL_Q_NORM) \
            else "k_norm.weight"
        return read_vector(source, f"{prefix}.{layer}.self_attn.{part}", head_dim)
    if kind == KIND_SLIDING_QUERY:
        rows = entry["rows"]
        return read_matrix(source, f"{prefix}.{layer}.self_attn.q_proj.weight",
                           geometry["sliding_q_heads"] * geometry["sliding_head_dim"],
                           hidden, tp_rank * rows, rows)
    if kind == KIND_SLIDING_KV:
        kv_heads, _ = kv_heads_per_rank(geometry["sliding_kv_heads"], tp_degree)
        head_rows = kv_heads * geometry["sliding_head_dim"]
        global_rows = geometry["sliding_kv_heads"] * geometry["sliding_head_dim"]
        k = read_matrix(source, f"{prefix}.{layer}.self_attn.k_proj.weight",
                        global_rows, hidden, tp_rank * head_rows, head_rows)
        v = read_matrix(source, f"{prefix}.{layer}.self_attn.v_proj.weight",
                        global_rows, hidden, tp_rank * head_rows, head_rows)
        return k + v
    if kind == KIND_SLIDING_OUTPUT:
        columns = entry["columns"]
        return read_matrix(source, f"{prefix}.{layer}.self_attn.o_proj.weight",
                           hidden, geometry["sliding_q_heads"] * geometry["sliding_head_dim"],
                           0, hidden, tp_rank * columns, columns)
    if kind == KIND_FULL_QUERY:
        rows = entry["rows"]
        return read_matrix(source, f"{prefix}.{layer}.self_attn.q_proj.weight",
                           geometry["full_q_heads"] * geometry["full_head_dim"],
                           hidden, tp_rank * rows, rows)
    if kind == KIND_FULL_KEY:
        kv_heads, replication = kv_heads_per_rank(geometry["full_kv_heads"], tp_degree)
        head = tp_rank // replication
        rows = kv_heads * geometry["full_head_dim"]
        global_rows = geometry["full_kv_heads"] * geometry["full_head_dim"]
        return read_matrix(source, f"{prefix}.{layer}.self_attn.k_proj.weight",
                           global_rows, hidden, head * geometry["full_head_dim"], rows)
    if kind == KIND_FULL_OUTPUT:
        columns = entry["columns"]
        return read_matrix(source, f"{prefix}.{layer}.self_attn.o_proj.weight",
                           hidden, geometry["full_q_heads"] * geometry["full_head_dim"],
                           0, hidden, tp_rank * columns, columns)
    if kind == KIND_MLP_GATE_UP:
        rows_per_branch = entry["rows"] // 2
        inter = geometry["dense_inter"]
        gate = read_projection_bf16(source, f"{prefix}.{layer}.mlp.gate_proj",
                                    inter, hidden, tp_rank * rows_per_branch,
                                    rows_per_branch)
        up = read_projection_bf16(source, f"{prefix}.{layer}.mlp.up_proj",
                                  inter, hidden, tp_rank * rows_per_branch,
                                  rows_per_branch)
        return gate + up
    if kind == KIND_MLP_DOWN:
        columns = entry["columns"]
        return read_projection_bf16(source, f"{prefix}.{layer}.mlp.down_proj",
                                    hidden, geometry["dense_inter"], 0, hidden,
                                    tp_rank * columns, columns)
    if kind == KIND_ROUTER_PROJ:
        proj = np.frombuffer(
            read_matrix(source, f"{prefix}.{layer}.router.proj.weight",
                        geometry["experts"], hidden), dtype="<u2").copy()
        scale_words = np.frombuffer(
            read_vector(source, f"{prefix}.{layer}.router.scale", hidden),
            dtype="<u2").astype("<u4") << 16
        scale = scale_words.view("<f4") * (hidden ** -0.5)
        values = (proj.reshape(geometry["experts"], hidden).astype("<u4") << 16).view("<f4")
        values *= scale[np.newaxis, :]
        proj = bf16_round_array(values.reshape(-1)).tobytes()
        return proj
    if kind == KIND_PER_EXPERT_SCALE:
        raw = read_vector(source, f"{prefix}.{layer}.router.per_expert_scale",
                          geometry["experts"])
        return bf16_blocks_to_f32(raw)
    if kind == KIND_EXPERT_GATE_UP:
        experts_per_rank = geometry["experts"] // tp_degree
        source_rows = 2 * geometry["expert_inter"]
        return read_matrix(
            source, f"{prefix}.{layer}.experts.gate_up_proj",
            geometry["experts"] * source_rows, hidden,
            tp_rank * experts_per_rank * source_rows, experts_per_rank * source_rows,
            expected_shape=[geometry["experts"], source_rows, hidden])
    if kind == KIND_EXPERT_DOWN:
        experts_per_rank = geometry["experts"] // tp_degree
        scaled = read_expert_down_folded(source, geometry, layer, tp_degree, tp_rank)
        if len(scaled) != entry["rows"] * entry["columns"] * 2:
            raise PackFailure(f"layer {layer} expert down fold size mismatch")
        return scaled
    raise PackFailure(f"payload_for: unhandled kind {kind}")


def bf16_round_array(values: "np.ndarray") -> "np.ndarray":
    """Round-to-nearest-even bf16 bit patterns for an f32 array."""
    bits = values.astype("<f4").view("<u4")
    high = bits >> 16
    low = bits & 0xFFFF
    round_up = ((low > 0x8000) | ((low == 0x8000) & ((high & 1) == 1))).astype("<u4")
    return ((high + round_up) & 0xFFFF).astype("<u2")


def bf16_round(value: float) -> int:
    packed = struct.unpack("<I", struct.pack("<f", value))[0]
    high = (packed >> 16) & 0xFFFF
    low = packed & 0xFFFF
    round_up = 1 if (low > 0x8000 or (low == 0x8000 and (high & 1) == 1)) else 0
    return (high + round_up) & 0xFFFF


def read_expert_down_folded(source: SafetensorsSource, geometry: dict, layer: int,
                            tp_degree: int, tp_rank: int) -> bytes:
    """Expert down rows with per_expert_scale folded in (contract pack fold)."""
    hidden, expert_inter = geometry["hidden"], geometry["expert_inter"]
    experts_per_rank = geometry["experts"] // tp_degree
    scale_raw = read_vector(source, f"model.language_model.layers.{layer}.router.per_expert_scale",
                            geometry["experts"])
    scales = []
    for e in range(geometry["experts"]):
        bits = scale_raw[2 * e] | (scale_raw[2 * e + 1] << 8)
        scales.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    name = f"model.language_model.layers.{layer}.experts.down_proj"
    shard, meta, data_start = source.resolve(name)
    if meta["shape"] != [geometry["experts"], hidden, expert_inter]:
        raise PackFailure(f"{name}: shape {meta['shape']}")
    block_bytes = hidden * expert_inter * 2
    pieces = []
    with (source.root / shard).open("rb") as file:
        for local in range(experts_per_rank):
            expert = tp_rank * experts_per_rank + local
            file.seek(data_start + expert * block_bytes)
            block = np.frombuffer(file.read(block_bytes), dtype="<u2").astype("<u4") << 16
            if block.size != hidden * expert_inter:
                raise PackFailure(f"{name}: short read")
            values = block.view("<f4") * scales[expert]
            pieces.append(bf16_round_array(values).tobytes())
    return b"".join(pieces)


MASK64 = (1 << 64) - 1
CK_C1 = 0x87c37b91114253d5
CK_C2 = 0x4cf5ad432745937f


def _rotl64(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & MASK64


def _fmix64(k: int) -> int:
    k ^= k >> 33
    k = (k * 0xff51afd7ed558ccd) & MASK64
    k ^= k >> 33
    k = (k * 0xc4ceb9fe1a85ec53) & MASK64
    k ^= k >> 33
    return k


def ck128(data: bytes) -> bytes:
    """Bit-exact port of src/spark_ck128.c (Murmur3-style 128-bit), which is
    what the weightd manifest parser verifies."""
    h1 = 0
    h2 = 0
    total = len(data)
    offset = 0
    aligned = total - (total % 16)
    for k1, k2 in struct.iter_unpack("<QQ", memoryview(data)[:aligned]):
        k1 = (k1 * CK_C1) & MASK64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * CK_C2) & MASK64
        h1 ^= k1
        h1 = (_rotl64(h1, 27) + h2) & MASK64
        h1 = (h1 * 5 + 0x52dce729) & MASK64
        k2 = (k2 * CK_C2) & MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * CK_C1) & MASK64
        h2 ^= k2
        h2 = (_rotl64(h2, 31) + h1) & MASK64
        h2 = (h2 * 5 + 0x38495ab5) & MASK64
    tail = data[aligned:]
    k1 = 0
    k2 = 0
    if len(tail) > 8:
        k2 = int.from_bytes(tail[8:16].ljust(8, b"\0"), "little")
        k2 = (k2 * CK_C2) & MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * CK_C1) & MASK64
        h2 ^= k2
    k1 = int.from_bytes(tail[:8].ljust(8, b"\0"), "little")
    k1 = (k1 * CK_C1) & MASK64
    k1 = _rotl64(k1, 31)
    k1 = (k1 * CK_C2) & MASK64
    h1 ^= k1
    h1 ^= total
    h2 ^= total
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    return struct.pack("<QQ", h1, h2)


def write_experts_manifest(source: SafetensorsSource, geometry: dict, plan: list[dict],
                           pack_path: Path, manifest_path: Path,
                           first_layer: int, layer_count: int,
                           tp_degree: int, tp_rank: int) -> dict:
    experts_per_rank = geometry["experts"] // tp_degree
    records = bytearray()
    records += struct.pack("<IIII", EXPERT_MANIFEST_MAGIC, EXPERT_MANIFEST_VERSION, 0, 0)
    record_count = 0
    expert_bytes_total = 0
    for entry in plan:
        if entry["kind"] not in (KIND_EXPERT_GATE_UP, KIND_EXPERT_DOWN):
            continue
        layer = entry["layer"]
        if not (first_layer <= layer < first_layer + layer_count):
            continue
        if entry["kind"] == KIND_EXPERT_GATE_UP:
            per_expert_rows = 2 * geometry["expert_inter"]
        else:
            per_expert_rows = geometry["hidden"]
        per_expert_bytes = per_expert_rows * entry["columns"] * 2
        with pack_path.open("rb") as pack_file:
            for local in range(experts_per_rank):
                offset = entry["payload_offset"] + local * per_expert_bytes
                pack_file.seek(offset)
                payload = pack_file.read(per_expert_bytes)
                if len(payload) != per_expert_bytes:
                    raise PackFailure("expert plane short read")
                records += EXPERT_RECORD_STRUCT.pack(
                    layer, tp_rank * experts_per_rank + local,
                    entry["kind"] * 2, 0, offset, per_expert_bytes, ck128(payload))
                record_count += 1
                expert_bytes_total += per_expert_bytes
    manifest_path.write_bytes(records)
    return dict(records=record_count, bytes=len(records),
                expert_bytes=expert_bytes_total)


def boundary_rank_checks(geometry: dict, tp_degree: int) -> list[dict]:
    """Frozen shard-map rulings, asserted as data so the receipt carries them."""
    checks = []
    sliding_heads_per_rank, sliding_rep = kv_heads_per_rank(
        geometry["sliding_kv_heads"], tp_degree)
    _, full_rep = kv_heads_per_rank(geometry["full_kv_heads"], tp_degree)
    for rank in (0, tp_degree - 1):
        checks.append(dict(
            rank=rank,
            sliding_kv_replication=sliding_rep,
            sliding_kv_source_head_base=rank * sliding_heads_per_rank,
            sliding_kv_heads_per_rank=sliding_heads_per_rank,
            full_kv_replication=full_rep,
            full_kv_source_head=rank // full_rep,
            embed_base_row=rank * (geometry["vocab"] // tp_degree),
        ))
    return checks


def verify_experts_manifest(manifest_path: Path, pack_path: Path, geometry: dict,
                            plan: list[dict], tp_degree: int, tp_rank: int,
                            first_layer: int, layer_count: int) -> dict:
    """Walk the placed experts manifest against the pack bytes: header, record
    count, per-record (layer, expert id, kind, offset, bytes) and the ck128
    payload digest. The expert id column is the rank-identity proof: a pack
    only matches the manifest written for its own tp_rank."""
    experts_per_rank = geometry["experts"] // tp_degree
    raw = manifest_path.read_bytes()
    magic, version, _zero0, _zero1 = struct.unpack("<IIII", raw[:16])
    report = dict(records=len(raw[16:]) // EXPERT_RECORD_STRUCT.size,
                  bytes=len(raw), magic_ok=magic == EXPERT_MANIFEST_MAGIC,
                  version_ok=version == EXPERT_MANIFEST_VERSION)
    problems: list[str] = []
    if not (report["magic_ok"] and report["version_ok"]):
        problems.append("experts manifest magic/version")
        report["problems"] = problems
        return report
    expected = {}
    record_count = 0
    for entry in plan:
        if entry["kind"] not in (KIND_EXPERT_GATE_UP, KIND_EXPERT_DOWN):
            continue
        if not (first_layer <= entry["layer"] < first_layer + layer_count):
            continue
        per_expert_rows = 2 * geometry["expert_inter"] \
            if entry["kind"] == KIND_EXPERT_GATE_UP else geometry["hidden"]
        expected[entry["layer"], entry["kind"]] = (
            entry["payload_offset"], per_expert_rows * entry["columns"] * 2)
        record_count += experts_per_rank
    if report["records"] != record_count:
        problems.append(f"experts manifest records {report['records']} != "
                        f"{record_count}")
        report["problems"] = problems
        return report
    id_base = tp_rank * experts_per_rank
    with pack_path.open("rb") as pack_file:
        for index in range(report["records"]):
            fields = EXPERT_RECORD_STRUCT.unpack_from(
                raw, 16 + index * EXPERT_RECORD_STRUCT.size)
            layer, expert, kind_code, _zero, offset, length, digest = fields
            key = (layer, kind_code // 2)
            if key not in expected:
                problems.append(f"record {index}: layer {layer} kind "
                                f"{kind_code} not in the stage plan")
                break
            base_offset, per_expert_bytes = expected[key]
            if length != per_expert_bytes:
                problems.append(f"record {index}: bytes {length} != "
                                f"{per_expert_bytes}")
                break
            if not id_base <= expert < id_base + experts_per_rank:
                problems.append(f"record {index}: expert {expert} outside the "
                                f"rank {tp_rank} span "
                                f"[{id_base}, {id_base + experts_per_rank})")
                break
            pack_file.seek(base_offset + (expert - id_base) * per_expert_bytes)
            payload = pack_file.read(per_expert_bytes)
            if len(payload) != per_expert_bytes:
                problems.append(f"record {index}: pack short read")
                break
            if ck128(payload)[:len(digest)] != digest:
                problems.append(f"record {index}: ck128 mismatch at layer "
                                f"{layer} expert {expert}")
                break
    report["problems"] = problems
    report["rank_span"] = [id_base, id_base + experts_per_rank]
    return report


def verify_existing(output: Path, geometry_name: str, first_layer: int,
                    layer_count: int, tp_degree: int,
                    tp_ranks: list[int]) -> dict:
    """Fast re-receipt for an already-packed stage: placement proof + manifest
    walk + spine/expert split, without touching the checkpoint. Each rank in
    tp_ranks is planned and compared against the pack's directory; with a
    single rank and a MoE geometry the experts manifest is walked with ck128
    digests and its expert-id span must match that rank."""
    geometry = GEOMETRY[geometry_name]
    if not tp_ranks:
        raise PackFailure("verify-existing needs at least one rank to plan")
    rank_reports = {}
    reference_plan = None
    header = None
    for tp_rank in sorted(set(tp_ranks)):
        plan = build_inventory(geometry, tp_degree, tp_rank, first_layer, layer_count)
        if reference_plan is None:
            reference_plan = plan
        for entry in plan:
            elements = entry["rows"] * entry["columns"]
            entry["payload_bytes"] = elements * (4 if entry["weight_format"] == WEIGHT_F32 else 2)
            entry["scale_bytes"] = 0
        plan, file_bytes, payload_bytes = place(plan)
        with output.open("rb") as pack:
            header = HEADER_STRUCT.unpack(pack.read(HEADER_BYTES))
            if header[0] != MAGIC or header[1] != FORMAT_VERSION:
                raise PackFailure("verify: bad magic/version")
            if header[4] != len(plan):
                raise PackFailure(f"verify: tensor count drift for the rank "
                                  f"{tp_rank} plan ({header[4]} on the wire, "
                                  f"{len(plan)} planned)")
            pack.seek(header[-2])
            directory = pack.read(header[4] * ENTRY_BYTES)
            mismatches = 0
            for index, entry in enumerate(plan):
                fields = ENTRY_STRUCT.unpack(
                    directory[index * ENTRY_BYTES:(index + 1) * ENTRY_BYTES])
                expected = (entry["kind"],
                            GLOBAL_LAYER if entry["layer"] == GLOBAL_LAYER else entry["layer"],
                            entry["weight_format"], entry["rows"], entry["columns"], 0,
                            entry["payload_offset"], entry["payload_bytes"], 0, 0)
                if fields != expected:
                    mismatches += 1
        rank_reports[tp_rank] = dict(
            passed=mismatches == 0, checked_entries=len(plan),
            mismatched_entries=mismatches, file_bytes=header[-1],
            directory_offset=header[-2])
    verified = dict(
        passed=all(report["passed"] for report in rank_reports.values()),
        ranks=rank_reports,
        checked_entries=len(reference_plan),
        file_bytes=header[-1], directory_offset=header[-2])
    manifest_path = Path(str(output).rsplit(".", 1)[0] + ".experts")
    manifest_present = manifest_path.is_file()
    if not manifest_present:
        alt = Path(str(output) + ".experts")
        if alt.is_file():
            manifest_path = alt
            manifest_present = True
    manifest_report = None
    if geometry["experts"] is not None and manifest_present:
        if len(rank_reports) != 1:
            raise PackFailure("verify: the experts manifest rank walk needs a "
                              "single-rank --verify-ranks list")
        manifest_report = verify_experts_manifest(
            manifest_path, output, geometry, reference_plan, tp_degree,
            next(iter(sorted(rank_reports))), first_layer, layer_count)
    expert_bytes = sum(e["payload_bytes"] for e in reference_plan
                       if e["kind"] in (KIND_EXPERT_GATE_UP, KIND_EXPERT_DOWN))
    return dict(model_id=geometry["model_id"], topology=geometry["topology"],
                tensor_count=len(reference_plan), file_bytes=header[-1],
                spine_bytes=payload_bytes - expert_bytes, expert_bytes=expert_bytes,
                experts_manifest=str(manifest_path) if manifest_present else None,
                experts_manifest_present=manifest_present,
                experts_manifest_records=None if manifest_report is None else manifest_report["records"],
                experts_manifest_ok=None if manifest_report is None else (manifest_report["magic_ok"] and manifest_report["version_ok"] and not manifest_report["problems"]),
                experts_manifest_problems=None if manifest_report is None else manifest_report["problems"],
                placement_proof=verified,
                boundary_ranks=boundary_rank_checks(geometry, tp_degree))


def convert(checkpoint: Path, output: Path, geometry_name: str, tp_degree: int,
            tp_rank: int, first_layer: int, layer_count: int,
            experts_manifest: Path | None) -> dict:
    geometry = GEOMETRY[geometry_name]
    source = SafetensorsSource(checkpoint, INDEX_NAME, CONFIG_NAME)
    hidden = geometry["hidden"]
    expectations = dict(hidden_size=hidden, num_hidden_layers=geometry["layers"],
                        vocab_size=geometry["vocab"])
    source.check_config(expectations, section="text_config")
    layer_scalars = {}
    for layer in range(first_layer, first_layer + layer_count):
        layer_scalars[layer] = read_layer_scalar(source, layer)
    plan = build_inventory(geometry, tp_degree, tp_rank, first_layer, layer_count)
    expected_count = expected_tensor_count(geometry, first_layer, layer_count)
    if len(plan) != expected_count:
        raise PackFailure(f"census locked: planned {len(plan)} tensors, "
                          f"expected {expected_count}")
    _, gate_meta, _ = source.resolve(
        f"model.language_model.layers.{first_layer}.mlp.gate_proj.weight")
    mlp_is_nvfp4 = gate_meta["dtype"] == "U8"
    if mlp_is_nvfp4:
        per_layer = {entry["kind"] for entry in plan}
        if KIND_EXPERT_GATE_UP in per_layer:
            raise PackFailure("the nvfp4 MLP reader covers the dense 31b arm "
                              "only; the MoE arm keeps native BF16 experts")
    plan, file_bytes, payload_bytes = place(plan)
    directory_offset = HEADER_BYTES
    fields = header_fields(geometry, plan, first_layer, layer_count, file_bytes,
                           directory_offset)
    directory = b"".join(ENTRY_STRUCT.pack(
        entry["kind"],
        GLOBAL_LAYER if entry["layer"] == GLOBAL_LAYER else entry["layer"],
        entry["weight_format"], entry["rows"], entry["columns"], 0,
        entry["payload_offset"], entry["payload_bytes"], 0, 0)
        for entry in plan)
    directory_sha = sha256_bytes(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as pack:
        pack.write(b"\0" * file_bytes)
        pack.seek(0)
        pack.write(HEADER_STRUCT.pack(*fields))
        pack.seek(directory_offset)
        pack.write(directory)
        for entry in plan:
            payload = payload_for(source, geometry, entry, tp_degree, tp_rank)
            if len(payload) != entry["payload_bytes"]:
                raise PackFailure(f"kind {entry['kind']} layer {entry['layer']}: "
                                  f"payload {len(payload)} bytes, planned {entry['payload_bytes']}")
            pack.seek(entry["payload_offset"])
            pack.write(payload)
    manifest_report = None
    if geometry["experts"] is not None:
        manifest_path = experts_manifest or output.with_suffix(".experts")
        manifest_report = write_experts_manifest(
            source, geometry, plan, output, manifest_path,
            first_layer, layer_count, tp_degree, tp_rank)
    verified = verify_pack(output, geometry, first_layer, layer_count,
                           directory_sha, plan)
    spine_bytes = payload_bytes - (manifest_report or dict(expert_bytes=0))["expert_bytes"]
    return dict(
        model_id=geometry["model_id"], topology=geometry["topology"],
        tp_degree=tp_degree, tp_rank=tp_rank,
        stage_layers=[first_layer, first_layer + layer_count],
        mlp_source=("nvfp4-decoded-bf16" if mlp_is_nvfp4 else "bf16-verbatim"),
        tensor_count=len(plan), census_expected=expected_count,
        file_bytes=file_bytes, payload_bytes=payload_bytes,
        spine_bytes=spine_bytes,
        layer_scalars=layer_scalars,
        expert_bytes=(manifest_report or dict(expert_bytes=0))["expert_bytes"],
        experts_manifest=None if manifest_report is None else str(manifest_path),
        experts_manifest_records=None if manifest_report is None else manifest_report["records"],
        directory_sha256=directory_sha.hex(),
        placement_proof=verified,
        boundary_ranks=boundary_rank_checks(geometry, tp_degree),
        source_sha256=source.index_sha256,
    )


def verify_pack(pack_path: Path, geometry: dict, first_layer: int, layer_count: int,
                directory_sha: str, plan: list[dict]) -> dict:
    """Pass 2: re-read the written pack and assert byte-exact placement."""
    with pack_path.open("rb") as pack:
        header = HEADER_STRUCT.unpack(pack.read(HEADER_BYTES))
        if header[0] != MAGIC or header[1] != FORMAT_VERSION:
            raise PackFailure("verify: bad magic/version")
        tensor_count, file_bytes = header[4], header[-1]
        if tensor_count != len(plan):
            raise PackFailure("verify: tensor count drift")
        pack.seek(header[-2])
        directory = pack.read(tensor_count * ENTRY_BYTES)
        if sha256_bytes(directory) != directory_sha:
            raise PackFailure("verify: directory sha drift")
        for index, entry in enumerate(plan):
            fields = ENTRY_STRUCT.unpack(
                directory[index * ENTRY_BYTES:(index + 1) * ENTRY_BYTES])
            if fields[6] != entry["payload_offset"] or fields[7] != entry["payload_bytes"]:
                raise PackFailure(f"verify: entry {index} placement drift")
            if fields[6] + fields[7] > file_bytes:
                raise PackFailure(f"verify: entry {index} exceeds file")
    return dict(passed=True, checked_entries=len(plan),
                file_bytes=header[-1], directory_offset=header[-2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(GEOMETRY), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-degree", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=None)
    parser.add_argument("--experts-manifest", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--verify-ranks", default=None,
        help="comma list of tp ranks to plan in --verify-existing mode "
             "(default: the single --tp-rank); the node's local rank pack "
             "is verified against its own rank's plan, and with a MoE "
             "geometry the experts manifest expert-id span + ck128 digests "
             "are proven for that rank")
    args = parser.parse_args()
    geometry = GEOMETRY[args.model]
    layer_count = args.layer_count
    if layer_count is None:
        layer_count = geometry["layers"]
    if args.verify_existing:
        if args.verify_ranks is None:
            tp_ranks = [args.tp_rank]
        else:
            tp_ranks = sorted({int(r) for r in args.verify_ranks.split(",")
                               if r != ""})
            if not tp_ranks:
                raise PackFailure("--verify-ranks is empty")
            if any(not 0 <= r < args.tp_degree for r in tp_ranks):
                raise PackFailure("--verify-ranks outside 0..tp-degree-1")
        receipt = verify_existing(args.output, args.model, args.first_layer,
                                  layer_count, args.tp_degree, tp_ranks)
        receipt["checkpoint"] = "existing pack (verify-only)"
        receipt["tool"] = "tools/gemma4_stagepack.py"
        receipt["verify_ranks"] = tp_ranks
        receipt_path = args.receipt or Path(str(args.output) + ".verify-receipt.json")
        write_receipt(receipt, receipt_path, suffix=None)
        proof = receipt["placement_proof"]
        print(f"gemma4_stagepack: verify-only {args.output} ranks={tp_ranks} "
              f"tensors={receipt['tensor_count']} "
              f"proof={proof['passed']} "
              f"manifest_ok={receipt['experts_manifest_ok']}")
        return 0 if proof["passed"] and receipt["experts_manifest_ok"] is not False else 1
    if args.tp_rank >= args.tp_degree:
        raise PackFailure("tp-rank out of range")
    if args.first_layer < 0 or args.first_layer + layer_count > geometry["layers"]:
        raise PackFailure(f"layer window [{args.first_layer}, "
                          f"{args.first_layer + layer_count}) outside the stack")

    receipt = convert(args.checkpoint, args.output, args.model, args.tp_degree,
                      args.tp_rank, args.first_layer, layer_count,
                      args.experts_manifest)
    receipt["checkpoint"] = str(args.checkpoint)
    receipt["tool"] = "tools/gemma4_stagepack.py"
    receipt_path = args.receipt or Path(str(args.output) + ".receipt.json")
    write_receipt(receipt, receipt_path, suffix=None)
    print(f"gemma4_stagepack: {args.output} tensors={receipt['tensor_count']} "
          f"file_bytes={receipt['file_bytes']} spine={receipt['spine_bytes']} "
          f"experts={receipt['expert_bytes']} proof={receipt['placement_proof']['passed']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as failure:
        print(f"SPARK_FAIL gemma4_stagepack: {failure}", file=sys.stderr)
        sys.exit(1)
