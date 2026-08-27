#!/usr/bin/env python3
"""Convert Qwen/Qwen3.8-2.4T-A95B safetensors checkpoints into qwen38 stage packs.

Setup-time code, never the serving path. Format v2: TP-sharded rank-local
packs (tp_degree/tp_rank in the 128-byte header, per-rank shapes from the
same shard-axis table the module's loader validates against).

Two checkpoint sources:

  --source-format fp8 (the vendor Qwen FP8 release): routed experts arrive
  F8_E4M3 with BF16 scale_inv companions and are packed as FP8_E4M3_F32B128.
  Non-expert tensors are BF16 and copied verbatim.

  --source-format quark-mxfp4 (amd/Qwen3.8-2.4T-A95B-Quark-MXFP4): routed
  experts arrive E2M1-packed U8 [rows, cols/2] with E8M0 group-32 scale
  companions U8 [rows, cols/32] - byte-identical to the module's
  MXFP4-E2M1 kernel layout - and are copied VERBATIM (professional
  calibration quantization, no requantization). Non-expert tensors stay
  BF16 (the Quark exclude list keeps the whole spine, router, shared
  expert, MTP and lm_head unquantized) and copy verbatim.

Shard axes (mirroring SparkQwen38MaxStagePackShardAxisOf): expert rows
(MOE W1/W3/DOWN), head rows (GDN QKV/conv composed q|k|v, gate, beta,
decay, ATTN q/k/v), input columns (GDN/ATTN output projections), head
columns (A_log/dt_bias). Everything else replicated. tp_degree 1 emits
the full-width v1-equivalent shapes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

# Make the sibling shared packer core importable however this tool is loaded.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from spark_pack_common import (  # noqa: E402
    PackFailure,
    SafetensorsSource as _BaseSafetensorsSource,
    align_up,
    sha256_file,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "model_contracts" / "qwen38_authoritative.json"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

# Wire constants, mirroring spark_qwen38_max_stagepack_format.h (v2).
MAGIC = 0x50533851  # 'Q8SP'
FORMAT_VERSION = 2
HEADER_BYTES = 128
ENTRY_BYTES = 56
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE
PAYLOAD_ALIGNMENT = 256

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_F32B128 = 4
WEIGHT_MXFP4_E2M1 = 7

HIDDEN = 8192
LAYER_COUNT = 92
ATTENTION_PERIOD = 4
FULL_PHASE = 3
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 128
GDN_HEAD_KEY_DIM = 128
GDN_HEAD_VALUE_DIM = 128
GDN_CONV_KERNEL = 4
ATTN_QUERY_HEADS = 64
ATTN_KV_HEADS = 4
ATTN_HEAD_DIM = 256
ATTN_ROPE_DIM = 64
EXPERT_COUNT = 512
EXPERTS_PER_TOKEN = 10
EXPERT_INTERMEDIATE = 2048
VOCAB = 248320
MTP_LAYERS = 1
MXFP4_GROUP = 32

GDN_QK_DIM = GDN_KEY_HEADS * GDN_HEAD_KEY_DIM            # 2048
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_VALUE_DIM     # 16384
GDN_CONV_CHANNELS = 2 * GDN_QK_DIM + GDN_VALUE_DIM       # 20480
ATTN_Q_DIM = ATTN_QUERY_HEADS * ATTN_HEAD_DIM            # 16384
ATTN_KV_DIM = ATTN_KV_HEADS * ATTN_HEAD_DIM              # 1024

HEADER_STRUCT = struct.Struct("<28I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES

# Tensor kinds, mirroring SparkQwen38MaxStagePackTensorKind.
(KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD, KIND_ATTENTION_NORM,
 KIND_MLP_NORM, KIND_MOE_GATE, KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN,
 KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_UP, KIND_MOE_SHARED_DOWN,
 KIND_MOE_SHARED_GATE_WEIGHT, KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_BETA,
 KIND_GDN_DECAY, KIND_GDN_OUTPUT, KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG,
 KIND_GDN_DT_BIAS, KIND_GDN_NORM, KIND_ATTN_QUERY, KIND_ATTN_KEY,
 KIND_ATTN_VALUE, KIND_ATTN_OUTPUT, KIND_ATTN_QUERY_NORM,
 KIND_ATTN_KEY_NORM, KIND_MTP_FC, KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM,
 KIND_MTP_FINAL_NORM) = range(32)

CHUNK_BYTES = 8 * 1024 * 1024
BF16_BYTES = 2
F32_BYTES = 4

MTP_PREFIX = "mtp."


def is_gdn_layer(layer_index: int) -> bool:
    return (layer_index % ATTENTION_PERIOD) != FULL_PHASE


# (rows, columns, weight_format) per kind at a TP degree. The shard axes are
# the C header's; the axis is applied to the FULL-width shape.
def kind_shape(kind: int, tp_degree: int = 1) -> tuple[int, int, int]:
    def rows_sharded(rows: int) -> int:
        return rows // tp_degree if kind in (
            KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN, KIND_GDN_QKV,
            KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY,
            KIND_GDN_CONV_WEIGHT, KIND_ATTN_QUERY, KIND_ATTN_KEY,
            KIND_ATTN_VALUE) else rows

    def cols_sharded(columns: int) -> int:
        return columns // tp_degree if kind in (
            KIND_GDN_OUTPUT, KIND_ATTN_OUTPUT, KIND_GDN_A_LOG,
            KIND_GDN_DT_BIAS) else columns

    table = {
        KIND_EMBEDDING: (VOCAB, HIDDEN, WEIGHT_BF16),
        KIND_FINAL_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_LM_HEAD: (VOCAB, HIDDEN, WEIGHT_BF16),
        KIND_ATTENTION_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MLP_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MOE_GATE: (EXPERT_COUNT, HIDDEN, WEIGHT_BF16),
    }
    if kind in table:
        return table[kind]
    expert_format = EXPERT_FORMAT[0]  # set by main() from --source-format
    if kind in (KIND_MOE_W1, KIND_MOE_W3):
        return (rows_sharded(EXPERT_COUNT * EXPERT_INTERMEDIATE), HIDDEN, expert_format)
    if kind == KIND_MOE_DOWN:
        return (rows_sharded(EXPERT_COUNT * HIDDEN), EXPERT_INTERMEDIATE, expert_format)
    if kind in (KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_UP):
        return (EXPERT_INTERMEDIATE, HIDDEN, WEIGHT_BF16)
    if kind == KIND_MOE_SHARED_DOWN:
        return (HIDDEN, EXPERT_INTERMEDIATE, WEIGHT_BF16)
    if kind == KIND_MOE_SHARED_GATE_WEIGHT:
        return (1, HIDDEN, WEIGHT_BF16)
    if kind == KIND_GDN_QKV:
        return (rows_sharded(GDN_CONV_CHANNELS), HIDDEN, WEIGHT_BF16)
    if kind == KIND_GDN_GATE:
        return (rows_sharded(GDN_VALUE_DIM), HIDDEN, WEIGHT_BF16)
    if kind in (KIND_GDN_BETA, KIND_GDN_DECAY):
        return (rows_sharded(GDN_VALUE_HEADS), HIDDEN, WEIGHT_BF16)
    if kind == KIND_GDN_OUTPUT:
        return (HIDDEN, cols_sharded(GDN_VALUE_DIM), WEIGHT_BF16)
    if kind == KIND_GDN_CONV_WEIGHT:
        return (rows_sharded(GDN_CONV_CHANNELS), GDN_CONV_KERNEL, WEIGHT_BF16)
    if kind == KIND_GDN_A_LOG:
        return (1, cols_sharded(GDN_VALUE_HEADS), WEIGHT_F32)
    if kind == KIND_GDN_DT_BIAS:
        return (1, cols_sharded(GDN_VALUE_HEADS), WEIGHT_F32)
    if kind == KIND_GDN_NORM:
        return (1, GDN_HEAD_VALUE_DIM, WEIGHT_BF16)
    if kind == KIND_ATTN_QUERY:
        return (rows_sharded(2 * ATTN_Q_DIM), HIDDEN, WEIGHT_BF16)
    if kind in (KIND_ATTN_KEY, KIND_ATTN_VALUE):
        return (rows_sharded(ATTN_KV_DIM), HIDDEN, WEIGHT_BF16)
    if kind == KIND_ATTN_OUTPUT:
        return (HIDDEN, cols_sharded(ATTN_Q_DIM), WEIGHT_BF16)
    if kind in (KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM):
        return (1, ATTN_HEAD_DIM, WEIGHT_BF16)
    if kind == KIND_MTP_FC:
        return (HIDDEN, 2 * HIDDEN, WEIGHT_BF16)
    if kind in (KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM):
        return (1, HIDDEN, WEIGHT_BF16)
    raise PackFailure(f"kind {kind} has no shape")


def layer_tensor_name(kind: int, layer: int) -> str:
    prefix = f"model.layers.{layer}." if layer != MTP_LAYER else MTP_PREFIX + "layers.0."
    gdn = {
        KIND_GDN_QKV: "linear_attn.in_proj_qkv.weight",
        KIND_GDN_GATE: "linear_attn.in_proj_z.weight",
        KIND_GDN_BETA: "linear_attn.in_proj_b.weight",
        KIND_GDN_DECAY: "linear_attn.in_proj_a.weight",
        KIND_GDN_OUTPUT: "linear_attn.out_proj.weight",
        KIND_GDN_CONV_WEIGHT: "linear_attn.conv1d.weight",
        KIND_GDN_A_LOG: "linear_attn.A_log",
        KIND_GDN_DT_BIAS: "linear_attn.dt_bias",
        KIND_GDN_NORM: "linear_attn.norm.weight",
    }
    attn = {
        KIND_ATTN_QUERY: "self_attn.q_proj.weight",
        KIND_ATTN_KEY: "self_attn.k_proj.weight",
        KIND_ATTN_VALUE: "self_attn.v_proj.weight",
        KIND_ATTN_OUTPUT: "self_attn.o_proj.weight",
        KIND_ATTN_QUERY_NORM: "self_attn.q_norm.weight",
        KIND_ATTN_KEY_NORM: "self_attn.k_norm.weight",
    }
    every = {
        KIND_ATTENTION_NORM: "input_layernorm.weight",
        KIND_MLP_NORM: "post_attention_layernorm.weight",
        KIND_MOE_GATE: "mlp.gate.weight",
        KIND_MOE_W1: "mlp.experts.{e}.gate_proj.weight",
        KIND_MOE_W3: "mlp.experts.{e}.up_proj.weight",
        KIND_MOE_DOWN: "mlp.experts.{e}.down_proj.weight",
        KIND_MOE_SHARED_GATE: "mlp.shared_expert.gate_proj.weight",
        KIND_MOE_SHARED_UP: "mlp.shared_expert.up_proj.weight",
        KIND_MOE_SHARED_DOWN: "mlp.shared_expert.down_proj.weight",
        KIND_MOE_SHARED_GATE_WEIGHT: "mlp.shared_expert_gate.weight",
    }
    for mapping in (every, gdn, attn):
        if kind in mapping:
            return prefix + mapping[kind]
    raise PackFailure(f"kind {kind} is not a per-layer tensor")


GLOBAL_TENSORS = {
    KIND_EMBEDDING: "model.embed_tokens.weight",
    KIND_FINAL_NORM: "model.norm.weight",
    KIND_LM_HEAD: "lm_head.weight",
    KIND_MTP_FC: MTP_PREFIX + "fc.weight",
    KIND_MTP_EMBED_NORM: MTP_PREFIX + "pre_fc_norm_embedding.weight",
    KIND_MTP_HIDDEN_NORM: MTP_PREFIX + "pre_fc_norm_hidden.weight",
    KIND_MTP_FINAL_NORM: MTP_PREFIX + "norm.weight",
}

EVERY_LAYER_KINDS = (KIND_ATTENTION_NORM, KIND_MLP_NORM, KIND_MOE_GATE,
                     KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN,
                     KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_UP,
                     KIND_MOE_SHARED_DOWN, KIND_MOE_SHARED_GATE_WEIGHT)
GDN_LAYER_KINDS = (KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY,
                   KIND_GDN_OUTPUT, KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG,
                   KIND_GDN_DT_BIAS, KIND_GDN_NORM)
ATTN_LAYER_KINDS = (KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE,
                    KIND_ATTN_OUTPUT, KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM)

# Module-level expert codec and rank, set from the CLI (kind_shape and the
# copy plans read them).
EXPERT_FORMAT = [WEIGHT_FP8_F32B128]
TP_RANK = [0]


class TensorRef:
    def __init__(self, kind: int, layer: int, name: str, tp_degree: int):
        self.kind = kind
        self.layer = layer
        self.name = name
        self.tp_degree = tp_degree
        self.rows, self.columns, self.weight_format = kind_shape(kind, tp_degree)
        self.slice_start, self.slice_rows = 0, 0


def expected_tensor_count(first_layer: int, layer_count: int) -> int:
    full_below = lambda n: n // ATTENTION_PERIOD
    full = full_below(first_layer + layer_count) - full_below(first_layer)
    gdn = layer_count - full
    tensors = layer_count * 10 + gdn * 9 + full * 6
    if first_layer == 0:
        tensors += 1
    if first_layer + layer_count == LAYER_COUNT:
        tensors += 2 + 4 + 16 + (1 if first_layer != 0 else 0)
    return tensors


def build_inventory(first_layer: int, layer_count: int, tp_degree: int) -> list[TensorRef]:
    if layer_count == 0 or first_layer + layer_count > LAYER_COUNT:
        raise PackFailure(f"invalid slice {first_layer}+{layer_count} of {LAYER_COUNT}")
    refs: list[TensorRef] = []
    if first_layer == 0:
        refs.append(TensorRef(KIND_EMBEDDING, GLOBAL_LAYER, GLOBAL_TENSORS[KIND_EMBEDDING], tp_degree))
    for layer in range(first_layer, first_layer + layer_count):
        class_kinds = GDN_LAYER_KINDS if is_gdn_layer(layer) else ATTN_LAYER_KINDS
        for kind in EVERY_LAYER_KINDS + class_kinds:
            refs.append(TensorRef(kind, layer, layer_tensor_name(kind, layer), tp_degree))
    if first_layer + layer_count == LAYER_COUNT:
        if first_layer != 0:
            refs.append(TensorRef(KIND_EMBEDDING, GLOBAL_LAYER, GLOBAL_TENSORS[KIND_EMBEDDING], tp_degree))
        for kind in (KIND_FINAL_NORM, KIND_LM_HEAD, KIND_MTP_FC,
                     KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM):
            refs.append(TensorRef(kind, GLOBAL_LAYER, GLOBAL_TENSORS[kind], tp_degree))
        for kind in EVERY_LAYER_KINDS + ATTN_LAYER_KINDS:
            refs.append(TensorRef(kind, MTP_LAYER, layer_tensor_name(kind, MTP_LAYER), tp_degree))
    expected = expected_tensor_count(first_layer, layer_count)
    if len(refs) != expected:
        raise PackFailure(f"inventory {len(refs)} tensors, format expects {expected}")
    return refs


class SafetensorsSource(_BaseSafetensorsSource):
    """qwen38 checkpoint reader: shared index/header/payload resolution plus
    the model's config expectations and per-source expert checks."""

    def check_config(self) -> None:
        expectations = {
            "hidden_size": HIDDEN, "num_hidden_layers": LAYER_COUNT,
            "num_attention_heads": ATTN_QUERY_HEADS,
            "num_key_value_heads": ATTN_KV_HEADS, "head_dim": ATTN_HEAD_DIM,
            "linear_num_key_heads": GDN_KEY_HEADS,
            "linear_num_value_heads": GDN_VALUE_HEADS,
            "linear_key_head_dim": GDN_HEAD_KEY_DIM,
            "linear_value_head_dim": GDN_HEAD_VALUE_DIM,
            "linear_conv_kernel_dim": GDN_CONV_KERNEL,
            "num_experts": EXPERT_COUNT,
            "num_experts_per_tok": EXPERTS_PER_TOKEN,
            "moe_intermediate_size": EXPERT_INTERMEDIATE,
            "vocab_size": VOCAB,
            "full_attention_interval": ATTENTION_PERIOD,
            "attn_output_gate": True, "tie_word_embeddings": False,
        }
        super().check_config(expectations)

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        if ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN):
            # per-expert tensors: validate expert 0 and its scale companion
            # (vendor FP8: F8_E4M3 + BF16 scale_inv; quark: U8 pairs + U8 e8m0).
            expert0 = ref.name.replace("{e}", "0")
            shard, meta, offset = self.resolve(expert0)
            if EXPERT_FORMAT[0] == WEIGHT_MXFP4_E2M1:
                if meta["dtype"] != "U8":
                    raise PackFailure(f"{expert0}: dtype {meta['dtype']}, expected U8 (E2M1 pairs)")
                if meta["shape"][1] * 2 != ref.columns:
                    raise PackFailure(f"{expert0}: packed width {meta['shape']} vs columns {ref.columns}")
                scale_name = expert0 + "_scale"
                _, scale_meta, _ = self.resolve(scale_name)
                if scale_meta["dtype"] != "U8" or scale_meta["shape"][1] != ref.columns // MXFP4_GROUP:
                    raise PackFailure(f"{scale_name}: unexpected scale shape {scale_meta['shape']}")
            else:
                if meta["dtype"] != "F8_E4M3":
                    raise PackFailure(f"{expert0}: dtype {meta['dtype']}, expected F8_E4M3")
                scale_name = expert0 + "_scale_inv"
                _, scale_meta, _ = self.resolve(scale_name)
                if scale_meta["dtype"] != "BF16":
                    raise PackFailure(f"{scale_name}: dtype {scale_meta['dtype']}, expected BF16")
            return shard, meta, offset
        # Non-expert: the checkpoint shape is the FULL-width tensor; the pack
        # slice is derived from the shard axes at copy time.
        return super().check_shape(ref.name, *self._full_shape(ref))

    def _full_shape(self, ref: TensorRef) -> tuple[int, int]:
        """The checkpoint-side (rows, columns) of a sharded non-expert ref."""
        if ref.tp_degree <= 1:
            return ref.rows, ref.columns
        if ref.kind in (KIND_GDN_A_LOG, KIND_GDN_DT_BIAS):
            return 1, ref.columns * ref.tp_degree
        if ref.kind in (KIND_GDN_OUTPUT, KIND_ATTN_OUTPUT):
            return ref.rows, ref.columns * ref.tp_degree
        return ref.rows * ref.tp_degree, ref.columns


# -- payload writers -----------------------------------------------------------


def copy_bf16_tensor(source: SafetensorsSource, ref: TensorRef, offset: int, out) -> None:
    """Copy a (possibly row- or column-sharded) BF16/F32 tensor, widening
    F32 for A_log/dt_bias as bf16 is the top half of f32."""
    path = source.root / source.weight_map[ref.name]
    elements = ref.rows * ref.columns
    f32 = ref.weight_format == WEIGHT_F32
    source_bytes = elements * BF16_BYTES  # source is BF16; f32 pack tensors widen
    with path.open("rb") as file:
        file.seek(offset)
        remaining = source_bytes
        while remaining > 0:
            step = min(remaining, CHUNK_BYTES)
            chunk = file.read(step)
            if len(chunk) != step:
                raise PackFailure(f"short read on {ref.name}")
            remaining -= step
            if not f32:
                out.write(chunk)
            else:  # f32: bf16 is the top half of f32
                widened = bytearray(step * 2)
                widened[2::4] = chunk[0::2]
                widened[3::4] = chunk[1::2]
                out.write(widened)


def sharded_bf16_plan(ref: TensorRef) -> tuple:
    """The rank's slice plan inside the full-width checkpoint tensor.

    Returns ("plain", row_start, row_count, col_start, col_count, full_cols)
    for row/column cuts, or ("segments", [(row_start, row_count), ...],
    full_cols) for the composed GDN q|k|v channel cut (the rank takes its
    whole-head slice of each third, kept contiguous in q|k|v order).
    """
    tp = ref.tp_degree
    rank = TP_RANK[0]
    if tp <= 1:
        return ("plain", 0, ref.rows, 0, ref.columns, ref.columns)
    if ref.kind in (KIND_GDN_OUTPUT, KIND_ATTN_OUTPUT, KIND_GDN_A_LOG, KIND_GDN_DT_BIAS):
        # input/head-column cut: the full width is ref.columns * tp.
        return ("plain", 0, ref.rows, rank * ref.columns, ref.columns, ref.columns * tp)
    if ref.kind in (KIND_GDN_QKV, KIND_GDN_CONV_WEIGHT):
        local_qk = GDN_QK_DIM // tp
        local_v = GDN_VALUE_DIM // tp
        segments = [(rank * local_qk, local_qk),
                    (GDN_QK_DIM + rank * local_qk, local_qk),
                    (2 * GDN_QK_DIM + rank * local_v, local_v)]
        return ("segments", segments, ref.columns)
    # head/expert row cut: the full height is ref.rows * tp.
    return ("plain", rank * ref.rows, ref.rows, 0, ref.columns, ref.columns)


def copy_sharded_bf16(source: SafetensorsSource, ref: TensorRef, offset: int, out) -> None:
    """Copy the rank's BF16 slice (F32 pack tensors widen from the BF16
    source, whose bytes are the top half of each f32)."""
    plan = sharded_bf16_plan(ref)
    path = source.root / source.weight_map[ref.name]
    widen = ref.weight_format == WEIGHT_F32
    with path.open("rb") as file:
        if plan[0] == "segments":
            _, segments, full_cols = plan
            for row_start, row_count in segments:
                file.seek(offset + (row_start * full_cols) * BF16_BYTES)
                _copy_rows(file, out, row_count, full_cols, ref.columns, widen)
        else:
            _, row_start, row_count, col_start, col_count, full_cols = plan
            file.seek(offset + ((row_start * full_cols) + col_start) * BF16_BYTES)
            _copy_rows(file, out, row_count, full_cols, col_count, widen)


def _copy_rows(file, out, row_count: int, full_cols: int, out_cols: int, widen: bool) -> None:
    """Copy row_count rows of out_cols BF16 elements from rows that are
    full_cols wide, streaming in chunks that never span a row boundary."""
    row_bytes = out_cols * BF16_BYTES
    skip_bytes = (full_cols - out_cols) * BF16_BYTES
    for _ in range(row_count):
        remaining = row_bytes
        while remaining > 0:
            step = min(remaining, CHUNK_BYTES)
            chunk = file.read(step)
            if len(chunk) != step:
                raise PackFailure("short read in sharded copy")
            if not widen:
                out.write(chunk)
            else:
                widened = bytearray(step * 2)
                widened[2::4] = chunk[0::2]
                widened[3::4] = chunk[1::2]
                out.write(widened)
            remaining -= step
        if skip_bytes:
            file.seek(skip_bytes, 1)


def copy_fp8_experts(source: SafetensorsSource, ref: TensorRef, out) -> None:
    """Stack per-expert F8_E4M3 weights and BF16 scale_inv planes into the
    pack: payload [E*R, C] expert-major, scales [E*R/128, C/128] as F32
    row-major (the FP8_E4M3_F32B128 kernel layout; scale_inv stored
    verbatim as the multiplier plane). The rank's expert slice only."""
    import numpy as np
    experts = EXPERT_COUNT // ref.tp_degree
    first_expert = TP_RANK[0] * experts
    rows_per_expert = ref.rows // experts
    scale_rows = rows_per_expert // 128
    scale_cols = ref.columns // 128
    payload = bytearray(ref.rows * ref.columns)
    scales = bytearray(ref.rows * ref.columns // (128 * 128) * 4)
    for e in range(experts):
        name = ref.name.replace("{e}", str(first_expert + e))
        shard, meta, off = source.resolve(name)
        with (source.root / shard).open("rb") as f:
            f.seek(off)
            raw = f.read(rows_per_expert * ref.columns)
        if len(raw) != rows_per_expert * ref.columns:
            raise PackFailure(f"short read on {name}")
        base = e * rows_per_expert * ref.columns
        payload[base:base + len(raw)] = raw
        scale_name = name + "_scale_inv"
        s_shard, s_meta, s_off = source.resolve(scale_name)
        with (source.root / s_shard).open("rb") as f:
            f.seek(s_off)
            sraw = f.read(scale_rows * scale_cols * 2)
        s16 = np.frombuffer(sraw, dtype="<u2").astype(np.uint32)
        s32 = ((s16 << 16).astype(np.uint32)).view(np.float32).astype("<f4").tobytes()
        sbase = e * scale_rows * scale_cols * 4
        scales[sbase:sbase + len(s32)] = s32
    out.write(payload)
    out.write(scales)


def copy_quark_mxfp4_experts(source: SafetensorsSource, ref: TensorRef, out) -> None:
    """Concatenate the rank's per-expert E2M1 payloads then their E8M0
    scale planes, verbatim: the checkpoint's [R, C/2] + [R, C/32] layout IS
    the pack's MXFP4 layout (byte-identical kernel decode)."""
    experts = EXPERT_COUNT // ref.tp_degree
    first_expert = TP_RANK[0] * experts
    rows_per_expert = ref.rows // experts
    payload_row_bytes = rows_per_expert * ref.columns // 2
    scale_row_bytes = rows_per_expert * ref.columns // MXFP4_GROUP
    payload = bytearray(payload_row_bytes * experts)
    scales = bytearray(scale_row_bytes * experts)
    for e in range(experts):
        name = ref.name.replace("{e}", str(first_expert + e))
        shard, meta, off = source.resolve(name)
        with (source.root / shard).open("rb") as f:
            f.seek(off)
            raw = f.read(payload_row_bytes)
        if len(raw) != payload_row_bytes:
            raise PackFailure(f"short read on {name}")
        payload[e * payload_row_bytes:(e + 1) * payload_row_bytes] = raw
        scale_name = name + "_scale"
        s_shard, s_meta, s_off = source.resolve(scale_name)
        with (source.root / s_shard).open("rb") as f:
            f.seek(s_off)
            sraw = f.read(scale_row_bytes)
        if len(sraw) != scale_row_bytes:
            raise PackFailure(f"short read on {scale_name}")
        scales[e * scale_row_bytes:(e + 1) * scale_row_bytes] = sraw
    out.write(payload)
    out.write(scales)


def convert(checkpoint: Path, output: Path, first_layer: int, layer_count: int,
            tp_degree: int, tp_rank: int, receipt: dict, dry_run: bool) -> dict:
    source = SafetensorsSource(checkpoint)
    source.check_config()
    refs = build_inventory(first_layer, layer_count, tp_degree)
    plans = []
    cursor = 0
    for ref in refs:
        shard, meta, offset = source.check_shape(ref)
        if ref.weight_format == WEIGHT_MXFP4_E2M1:
            payload_bytes = ref.rows * ref.columns // 2
            scale_bytes = ref.rows * ref.columns // MXFP4_GROUP
        elif ref.weight_format == WEIGHT_FP8_F32B128:
            payload_bytes = ref.rows * ref.columns
            scale_bytes = (ref.rows // 128) * (ref.columns // 128) * F32_BYTES
        else:
            payload_bytes = ref.rows * ref.columns * (BF16_BYTES if ref.weight_format == WEIGHT_BF16 else F32_BYTES)
            scale_bytes = 0
        payload_offset = align_up(cursor, PAYLOAD_ALIGNMENT)
        plans.append((ref, offset, payload_offset, payload_bytes, scale_bytes))
        cursor = payload_offset + payload_bytes + scale_bytes
    payload_base = align_up(HEADER_BYTES + len(plans) * ENTRY_BYTES, PAYLOAD_ALIGNMENT)
    file_bytes = payload_base + cursor

    header = HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(plans),
        HIDDEN, layer_count, first_layer, LAYER_COUNT,
        ATTENTION_PERIOD, FULL_PHASE,
        GDN_KEY_HEADS, GDN_VALUE_HEADS, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM,
        GDN_CONV_KERNEL, ATTN_QUERY_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM,
        ATTN_ROPE_DIM, EXPERT_COUNT, EXPERTS_PER_TOKEN, EXPERT_INTERMEDIATE,
        VOCAB, MXFP4_GROUP, MTP_LAYERS, tp_degree, tp_rank,
        HEADER_BYTES, file_bytes)
    entries = b"".join(
        ENTRY_STRUCT.pack(
            ref.kind, ref.layer, ref.weight_format, ref.rows, ref.columns,
            (MXFP4_GROUP if ref.weight_format == WEIGHT_MXFP4_E2M1 else
             128 if ref.weight_format == WEIGHT_FP8_F32B128 else 0),
            payload_base + payload_offset, payload_bytes,
            payload_base + payload_offset + payload_bytes if scale_bytes else 0,
            scale_bytes)
        for ref, _, payload_offset, payload_bytes, scale_bytes in plans)
    receipt.update({
        "first_layer_index": first_layer,
        "layer_count": layer_count,
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "tensor_count": len(plans),
        "bytes": file_bytes,
        "source_index_sha256": source.index_sha256,
        "source_config_sha256": source.config_sha256,
    })
    if dry_run:
        print(f"qwen38_stagepack slice={first_layer}+{layer_count} tp={tp_degree}/{tp_rank} "
              f"tensors={len(plans)} file_bytes={file_bytes} "
              f"file_gib={file_bytes / 2**30:.2f} (dry run)")
        return receipt

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp",
                                     dir=output.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        temp.write(header)
        temp.write(entries)
        padding = payload_base - temp.tell()
        if padding < 0:
            raise PackFailure("directory overruns the payload base")
        temp.write(b"\0" * padding)
        for ref, source_offset, payload_offset, payload_bytes, scale_bytes in plans:
            before = temp.tell()
            if ref.weight_format == WEIGHT_MXFP4_E2M1:
                copy_quark_mxfp4_experts(source, ref, temp)
            elif ref.weight_format == WEIGHT_FP8_F32B128:
                copy_fp8_experts(source, ref, temp)
            elif ref.tp_degree > 1:
                copy_sharded_bf16(source, ref, source_offset, temp)
            else:
                copy_bf16_tensor(source, ref, source_offset, temp)
            wrote = temp.tell() - before
            if wrote != payload_bytes + scale_bytes:
                raise PackFailure(f"payload size mismatch on {ref.name}: {wrote} != {payload_bytes + scale_bytes}")
            pad = align_up(temp.tell(), PAYLOAD_ALIGNMENT) - temp.tell()
            if pad:
                temp.write(b"\0" * pad)
        temp.flush()
        os.fsync(temp.fileno())
    os.replace(temp_path, output)
    receipt["output_sha256"] = sha256_file(output)
    receipt["file"] = str(output)
    print(f"qwen38_stagepack slice={first_layer}+{layer_count} tp={tp_degree}/{tp_rank} tensors={len(plans)} "
          f"file_gib={file_bytes / 2**30:.2f} wrote {output}")
    return receipt


def main() -> int:
    global TP_RANK
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint directory")
    parser.add_argument("--output", type=Path, help="pack output path")
    parser.add_argument("--first-layer", type=int, help="explicit slice start")
    parser.add_argument("--layer-count", type=int, help="explicit slice length")
    parser.add_argument("--tp-degree", type=int, default=1, help="tensor-parallel degree (1, 2 or 4)")
    parser.add_argument("--tp-rank", type=int, default=0, help="this pack's tensor-parallel rank")
    parser.add_argument("--source-format", choices=("fp8", "quark-mxfp4"), default="fp8",
                        help="expert codec of the source checkpoint")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--receipt", type=Path, help="receipt output (default: <output>.receipt.json)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.first_layer is None or args.layer_count is None:
        parser.error("--first-layer and --layer-count are required")
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run")
    EXPERT_FORMAT[0] = WEIGHT_MXFP4_E2M1 if args.source_format == "quark-mxfp4" else WEIGHT_FP8_F32B128
    TP_RANK[0] = args.tp_rank
    if args.tp_degree not in (1, 2, 4) or args.tp_rank >= args.tp_degree:
        parser.error(f"invalid tp {args.tp_rank}/{args.tp_degree}: degree in {{1,2,4}}, rank < degree")
    for axis in (EXPERT_COUNT, GDN_VALUE_HEADS, GDN_KEY_HEADS, ATTN_QUERY_HEADS, ATTN_KV_HEADS):
        if axis % args.tp_degree != 0:
            parser.error(f"tp degree {args.tp_degree} does not shard {axis} evenly")

    receipt = {
        "kind": "sparkpipe.qwen38.stagepack-receipt.v2",
        "tool": "tools/qwen38_stagepack.py",
        "checkpoint": str(args.checkpoint),
        "contract": {"path": str(args.contract),
                     "sha256": sha256_file(args.contract) if args.contract.is_file() else None},
        "source_format": args.source_format,
        "weight_formats": {"routed_experts": "mxfp4_e2m1" if args.source_format == "quark-mxfp4" else "fp8_e4m3_f32b128",
                           "non_expert": "bf16",
                           "gdn_a_log_dt_bias": "f32"},
    }
    result = convert(args.checkpoint, args.output or Path("/dev/null"),
                     args.first_layer, args.layer_count, args.tp_degree, args.tp_rank,
                     receipt, args.dry_run)
    if not args.dry_run:
        receipt_path = args.receipt or Path(str(args.output) + ".receipt.json")
        write_receipt(result, receipt_path, suffix=None)
        print(f"qwen38_stagepack receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen38_stagepack: {error}", file=sys.stderr)
        sys.exit(1)
