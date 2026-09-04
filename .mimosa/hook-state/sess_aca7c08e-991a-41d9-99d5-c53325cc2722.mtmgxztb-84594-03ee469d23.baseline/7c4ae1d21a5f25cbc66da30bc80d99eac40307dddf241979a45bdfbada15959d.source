#!/usr/bin/env python3
"""Convert the Qwen/Qwen3.6-27B BF16 safetensors checkpoint into qwen38_27b stage packs.

Setup-time code, never the serving path: reads safetensors shard headers and
streams payloads into the wire format of
modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_stagepack_format.h.

The checkpoint layout was pinned against transformers modeling_qwen3_5 and
verified against the shard headers on disk:

  * GDN layers carry in_proj_qkv (ONE fused tensor, conv channel order
    q 2048 | k 2048 | v 6144), in_proj_z (the output gate), and SEPARATE
    in_proj_b / in_proj_a 48-row projections (beta and decay - the pack's
    GDN_BETA / GDN_DECAY kinds map to them directly, no split).
  * Attention q_proj is [12288, 5120]: each head's 512 rows are 256 query
    then 256 gate, the fused output gate. The pack copies it verbatim; the
    module consumes the fused layout.
  * A_log and dt_bias are BF16 in the checkpoint but F32 in the pack (the
    format header's natural format for both); the converter upcasts.
  * conv1d.weight is [10240, 1, 4]; the singleton dim drops, the payload is
    already contiguous [10240, 4].
  * The vision tower (model.visual.*) is out of scope by contract and never
    referenced; the MTP decoder layer reuses the per-layer kinds at the
    reserved MTP layer marker.
"""

from __future__ import annotations

import argparse
import json
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
    spark_pack_replicated_draft_rows,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "model_contracts" / "qwen38_27b_authoritative.json"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

# Wire constants, mirroring spark_qwen38_27b_stagepack_format.h. The round-trip
# test cross-checks these against the header so the two cannot drift apart.
MAGIC = 0x50533651  # 'Q6SP'
FORMAT_VERSION = 3
HEADER_BYTES = 120
ENTRY_BYTES = 56
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE
PAYLOAD_ALIGNMENT = 256

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_E4M3_F32B128 = 5
WEIGHT_NVFP4_PACKED = 8
NVFP4_GROUP = 16
FP8_SCALE_GROUP = 128  # FP8 per-128x128-block F32 scale (kernel SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128)

HIDDEN = 5120
LAYER_COUNT = 64
ATTENTION_PERIOD = 4
FULL_PHASE = 3
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_HEAD_KEY_DIM = 128
GDN_HEAD_VALUE_DIM = 128
GDN_CONV_KERNEL = 4
ATTN_QUERY_HEADS = 24
ATTN_KV_HEADS = 4
ATTN_HEAD_DIM = 256
ATTN_ROPE_DIM = 64
FFN_INTERMEDIATE = 17408
VOCAB = 248320
MTP_LAYERS = 1
MXFP4_GROUP = 32

GDN_QK_DIM = GDN_KEY_HEADS * GDN_HEAD_KEY_DIM          # 2048
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_VALUE_DIM   # 6144
GDN_CONV_CHANNELS = 2 * GDN_QK_DIM + GDN_VALUE_DIM     # 10240
ATTN_Q_DIM = ATTN_QUERY_HEADS * ATTN_HEAD_DIM          # 6144
ATTN_KV_DIM = ATTN_KV_HEADS * ATTN_HEAD_DIM            # 1024

HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES


# Tensor kinds, mirroring SparkQwen38_27bStagePackTensorKind.
(KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD, KIND_ATTENTION_NORM,
 KIND_MLP_NORM, KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN, KIND_GDN_QKV,
 KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY, KIND_GDN_OUTPUT,
 KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG, KIND_GDN_DT_BIAS, KIND_GDN_NORM,
 KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT,
 KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM, KIND_MTP_FC, KIND_MTP_EMBED_NORM,
 KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM) = range(27)

CHUNK_BYTES = 8 * 1024 * 1024
BF16_BYTES = 2
F32_BYTES = 4

LANGUAGE_PREFIX = "model.language_model."
MTP_PREFIX = "mtp."


def align(offset: int) -> int:
    return align_up(offset, PAYLOAD_ALIGNMENT)


def is_gdn_layer(layer_index: int) -> bool:
    return (layer_index % ATTENTION_PERIOD) != FULL_PHASE


# (rows, columns, weight_format) per kind, mirroring the Shape* functions in
# the format header. Norms are 1 x width rows; only A_log and dt_bias are f32.
def kind_shape(kind: int) -> tuple[int, int, int]:
    table = {
        KIND_EMBEDDING: (VOCAB, HIDDEN, WEIGHT_BF16),
        KIND_FINAL_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_LM_HEAD: (VOCAB, HIDDEN, WEIGHT_BF16),
        KIND_ATTENTION_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MLP_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_FFN_GATE: (FFN_INTERMEDIATE, HIDDEN, WEIGHT_BF16),
        KIND_FFN_UP: (FFN_INTERMEDIATE, HIDDEN, WEIGHT_BF16),
        KIND_FFN_DOWN: (HIDDEN, FFN_INTERMEDIATE, WEIGHT_BF16),
        KIND_GDN_QKV: (GDN_CONV_CHANNELS, HIDDEN, WEIGHT_BF16),
        KIND_GDN_GATE: (GDN_VALUE_DIM, HIDDEN, WEIGHT_BF16),
        KIND_GDN_BETA: (GDN_VALUE_HEADS, HIDDEN, WEIGHT_BF16),
        KIND_GDN_DECAY: (GDN_VALUE_HEADS, HIDDEN, WEIGHT_BF16),
        KIND_GDN_OUTPUT: (HIDDEN, GDN_VALUE_DIM, WEIGHT_BF16),
        KIND_GDN_CONV_WEIGHT: (GDN_CONV_CHANNELS, GDN_CONV_KERNEL, WEIGHT_BF16),
        KIND_GDN_A_LOG: (1, GDN_VALUE_HEADS, WEIGHT_F32),
        KIND_GDN_DT_BIAS: (1, GDN_VALUE_HEADS, WEIGHT_F32),
        KIND_GDN_NORM: (1, GDN_HEAD_VALUE_DIM, WEIGHT_BF16),
        KIND_ATTN_QUERY: (2 * ATTN_Q_DIM, HIDDEN, WEIGHT_BF16),
        KIND_ATTN_KEY: (ATTN_KV_DIM, HIDDEN, WEIGHT_BF16),
        KIND_ATTN_VALUE: (ATTN_KV_DIM, HIDDEN, WEIGHT_BF16),
        KIND_ATTN_OUTPUT: (HIDDEN, ATTN_Q_DIM, WEIGHT_BF16),
        KIND_ATTN_QUERY_NORM: (1, ATTN_HEAD_DIM, WEIGHT_BF16),
        KIND_ATTN_KEY_NORM: (1, ATTN_HEAD_DIM, WEIGHT_BF16),
        KIND_MTP_FC: (HIDDEN, 2 * HIDDEN, WEIGHT_BF16),
        KIND_MTP_EMBED_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MTP_HIDDEN_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MTP_FINAL_NORM: (1, HIDDEN, WEIGHT_BF16),
    }
    return table[kind]


# Checkpoint tensor name for a kind at a layer, or None for globals (named in
# the inventory builder). layer is a stack index or the MTP marker.
def layer_tensor_name(kind: int, layer: int) -> str:
    prefix = (LANGUAGE_PREFIX + f"layers.{layer}.") if layer != MTP_LAYER else (MTP_PREFIX + "layers.0.")
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
        KIND_FFN_GATE: "mlp.gate_proj.weight",
        KIND_FFN_UP: "mlp.up_proj.weight",
        KIND_FFN_DOWN: "mlp.down_proj.weight",
    }
    for mapping in (every, gdn, attn):
        if kind in mapping:
            return prefix + mapping[kind]
    raise PackFailure(f"kind {kind} is not a per-layer tensor")


GLOBAL_TENSORS = {
    KIND_EMBEDDING: LANGUAGE_PREFIX + "embed_tokens.weight",
    KIND_FINAL_NORM: LANGUAGE_PREFIX + "norm.weight",
    KIND_LM_HEAD: "lm_head.weight",
    KIND_MTP_FC: MTP_PREFIX + "fc.weight",
    KIND_MTP_EMBED_NORM: MTP_PREFIX + "pre_fc_norm_embedding.weight",
    KIND_MTP_HIDDEN_NORM: MTP_PREFIX + "pre_fc_norm_hidden.weight",
    KIND_MTP_FINAL_NORM: MTP_PREFIX + "norm.weight",
}

EVERY_LAYER_KINDS = (KIND_ATTENTION_NORM, KIND_MLP_NORM, KIND_FFN_GATE,
                     KIND_FFN_UP, KIND_FFN_DOWN)
GDN_LAYER_KINDS = (KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY,
                   KIND_GDN_OUTPUT, KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG,
                   KIND_GDN_DT_BIAS, KIND_GDN_NORM)
ATTN_LAYER_KINDS = (KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE,
                    KIND_ATTN_OUTPUT, KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM)


FP8_KINDS = frozenset([
    KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN,
    KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_OUTPUT,
    KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT,
])


class TensorRef:
    """One pack tensor: its kind, layer marker, and checkpoint source."""

    def __init__(self, kind: int, layer: int, name: str):
        self.kind = kind
        self.layer = layer
        self.name = name
        self.rows, self.columns, self.weight_format = kind_shape(kind)
        self.scale_name = None
        self.fused_up_offset = 0  # >0 under the fused gate|up source


def build_inventory(first_layer: int, layer_count: int) -> list[TensorRef]:
    """The slice's tensor list, in the synthesizer's emission order."""
    if layer_count == 0 or first_layer + layer_count > LAYER_COUNT:
        raise PackFailure(f"invalid slice {first_layer}+{layer_count} of {LAYER_COUNT}")
    refs: list[TensorRef] = []
    if first_layer == 0:
        refs.append(TensorRef(KIND_EMBEDDING, GLOBAL_LAYER, GLOBAL_TENSORS[KIND_EMBEDDING]))
    for layer in range(first_layer, first_layer + layer_count):
        class_kinds = GDN_LAYER_KINDS if is_gdn_layer(layer) else ATTN_LAYER_KINDS
        for kind in EVERY_LAYER_KINDS + class_kinds:
            refs.append(TensorRef(kind, layer, layer_tensor_name(kind, layer)))
    if first_layer + layer_count == LAYER_COUNT:
        # The head stage's MTP chain embeds its own draft tokens and the
        # vocabulary is untied: a second embedding copy unless this pack
        # already carries it as the stage-zero global.
        if first_layer != 0:
            refs.append(TensorRef(KIND_EMBEDDING, GLOBAL_LAYER, GLOBAL_TENSORS[KIND_EMBEDDING]))
        for kind in (KIND_FINAL_NORM, KIND_LM_HEAD, KIND_MTP_FC,
                     KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM):
            refs.append(TensorRef(kind, GLOBAL_LAYER, GLOBAL_TENSORS[kind]))
        for kind in EVERY_LAYER_KINDS + ATTN_LAYER_KINDS:
            refs.append(TensorRef(kind, MTP_LAYER, layer_tensor_name(kind, MTP_LAYER)))
    expected = expected_tensor_count(first_layer, layer_count)
    if len(refs) != expected:
        raise PackFailure(f"inventory {len(refs)} tensors, format expects {expected}")
    return refs


def expected_tensor_count(first_layer: int, layer_count: int) -> int:
    """SparkQwen38_27bStagePackExpectedTensorCount, restated."""
    full_below = lambda n: n // ATTENTION_PERIOD
    full = full_below(first_layer + layer_count) - full_below(first_layer)
    gdn = layer_count - full
    tensors = layer_count * 5 + gdn * 9 + full * 6
    if first_layer == 0:
        tensors += 1
    if first_layer + layer_count == LAYER_COUNT:
        tensors += 2 + 4 + 11 + (1 if first_layer != 0 else 0)
    return tensors


class SafetensorsSource(_BaseSafetensorsSource):
    """The checkpoint's shards: shared index/header/payload resolution plus
    the model's text_config expectations and per-layer type checks."""

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
            "intermediate_size": FFN_INTERMEDIATE, "vocab_size": VOCAB,
            "full_attention_interval": ATTENTION_PERIOD,
            "attn_output_gate": True, "tie_word_embeddings": False,
        }
        super().check_config(expectations, section="text_config")
        text = self.config.get("text_config", {})
        layer_types = text.get("layer_types", [])
        if len(layer_types) == LAYER_COUNT:
            for layer, layer_type in enumerate(layer_types):
                want = "linear_attention" if is_gdn_layer(layer) else "full_attention"
                if layer_type != want:
                    raise PackFailure(f"config layer_types[{layer}]={layer_type!r}, expected {want!r}")

    def check_shape_nvfp4a16(self, ref: TensorRef):
        """The nvfp4a16 release: main-layer FFN projections arrive as
        fused per-proj U8-packed e2m1 tensors with per-row x per-16
        e4m3 scale planes and ONE F32 weight_global_scale per tensor -
        weight-only quantization for the BF16-activation (a16) spine.
        The pack segment = [plane rows x cols/16][global F32]."""
        proj = {KIND_FFN_GATE: "gate_proj", KIND_FFN_UP: "up_proj",
                KIND_FFN_DOWN: "down_proj"}[ref.kind]
        stem = ref.name[:-len(".weight")]
        shard, meta, offset = self.resolve(stem + ".weight_packed")
        if meta["dtype"] != "U8" or meta["shape"] != [ref.rows, ref.columns // 2]:
            raise PackFailure(f"{stem}.weight_packed: {meta['dtype']} {meta['shape']}, expected U8 [{ref.rows},{ref.columns // 2}]")
        s_meta = self.resolve(stem + ".weight_scale")[1]
        if s_meta["dtype"] != "F8_E4M3" or s_meta["shape"] != [ref.rows, ref.columns // 16]:
            raise PackFailure(f"{stem}.weight_scale: {s_meta['dtype']} {s_meta['shape']}, expected F8_E4M3 [{ref.rows},{ref.columns // 16}]")
        g_meta = self.resolve(stem + ".weight_global_scale")[1]
        if g_meta["dtype"] != "F32" or g_meta["shape"] not in ([], [1]):
            raise PackFailure(f"{stem}.weight_global_scale: {g_meta['dtype']} {g_meta['shape']}, expected F32 scalar")
        ref.weight_format = WEIGHT_NVFP4_PACKED
        ref.scale_name = stem + ".weight_scale"
        ref.nvfp4_global_name = stem + ".weight_global_scale"
        return shard, meta, offset

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        # DTYPE-DRIVEN: read the tensor's actual dtype and select FP8 only when
        # it is F8_E4M3; every other tensor stays its natural BF16/F32 format
        # with no scale entry. The dtype passed down is the ACTUAL dtype, so
        # this verifies the checkpoint rather than forcing an expectation.
        _, meta, _ = self.resolve(ref.name)
        actual_dtype = meta["dtype"]
        # THE FUSED gate|up SOURCE (the official -fp8 release): one tensor
        # mlp.gate_proj.weight [2*I, H] carrying gate rows [0:I) then up
        # rows [I:2I); up_proj is absent. Probe once per ref and remap UP
        # onto the same tensor at the +I row offset (spec: coordinator-log
        # 2026-08-30 ~03:5x).
        if ref.kind in (KIND_FFN_GATE, KIND_FFN_UP):
            _, g_meta, _ = self.resolve(
                ref.name.rsplit(".up_proj.", 1)[0] + ".gate_proj.weight"
                if ref.kind == KIND_FFN_UP else ref.name)
            if g_meta["shape"][0] == 2 * FFN_INTERMEDIATE:
                ref.fused_up_offset = FFN_INTERMEDIATE if ref.kind == KIND_FFN_UP else 0
                ref.name = (ref.name.rsplit(".up_proj.", 1)[0] + ".gate_proj.weight"
                            if ref.kind == KIND_FFN_UP else ref.name)
        if ref.kind in FP8_KINDS and actual_dtype == "F8_E4M3":
            ref.weight_format = WEIGHT_FP8_E4M3_F32B128
            ref.scale_name = ref.name + "_scale_inv"
        else:
            ref.scale_name = None
        return super().check_shape(ref.name, ref.rows, ref.columns, dtype=actual_dtype)


# Qwen3_5RMSNorm applies x * (1 + weight) with zero-initialized weights and
# the checkpoint stores the raw weight. The pack stores the EFFECTIVE gain,
# so these kinds fold the +1 at pack time. The gated GDN norm (weight
# initialized to ones, applied directly) is NOT folded.
NORM_PLUS_ONE_KINDS = frozenset([
    KIND_ATTENTION_NORM, KIND_MLP_NORM, KIND_FINAL_NORM,
    KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM,
    KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM,
])

def bf16_u16_to_f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]

def f32_to_bf16_u16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


# ---------------------------------------------------------------------------
# Tensor-parallel sharding (recipe qwen38_27b.TP4). A TP pack stores ONE rank's
# row/column window of every shardable tensor, stitched where the checkpoint
# layout fuses several shard classes into one tensor. Replicated tensors
# (norms, conv, gates, beta/decay, MTP) are stored whole.
# ---------------------------------------------------------------------------

class Nvfp4A16Source(SafetensorsSource):
    """The qwen3.8-27b-nvfp4a16-bf16-spine release: the main-layer FFN
    projections are weight-only NVFP4 (U8-packed e2m1 + per-16 e4m3
    planes + one F32 weight_global_scale each); everything else - and
    the MTP block - stays plain BF16."""

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        if ref.kind in (KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN) and ref.layer != MTP_LAYER:
            return self.check_shape_nvfp4a16(ref)
        return super().check_shape(ref)


class TpSlice:
    """A contiguous window of a row-major [rows, columns] tensor."""
    def __init__(self, row_off, row_count, col_off, col_count):
        self.row_off = row_off
        self.row_count = row_count
        self.col_off = col_off
        self.col_count = col_count

class TpFusedSlice:
    """Row windows of several source row ranges stitched into one entry."""
    def __init__(self, segments):
        self.segments = segments          # [(row_off, row_count), ...]
        self.col_count = HIDDEN

def tp_window(total, degree, rank):
    if total % degree != 0:
        raise PackFailure(f"dimension {total} does not divide TP degree {degree}")
    width = total // degree
    return (rank * width, width)

# The qwen MTP replicated-draft table: the fc + three norms live at the
# GLOBAL layer and replicate; the MTP decoder's per-layer kinds slice.
QWEN_MTP_KINDS = frozenset((KIND_MTP_FC, KIND_MTP_EMBED_NORM,
                            KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM))


def build_tp_plan(ref, degree, rank):
    """Return (row_slice, col_slice, packed_rows, packed_cols). None = replicated."""
    if degree <= 1:
        return None
    if spark_pack_replicated_draft_rows(
            ref.kind, ref.layer, draft_layer_first=MTP_LAYER,
            draft_layer_count=MTP_LAYERS, global_kinds=QWEN_MTP_KINDS,
            draft_layer_kinds_slice=True):
        return None
    if ref.layer == GLOBAL_LAYER:
        if ref.kind == KIND_LM_HEAD:
            off, count = tp_window(VOCAB, degree, rank)
            return TpSlice(off, count, 0, HIDDEN)
        if ref.kind == KIND_EMBEDDING:
            return None  # replicated: no collective broadcast yet; the
                        # gather path reads the full table on every rank
        return None
    # The MTP decoder reuses the per-layer kinds: attention and FFN tensors
    # slice exactly like main layers (the fc and the three norms live at the
    # GLOBAL layer and replicate). Fall through to the kind-based plan below.
    if ref.kind in (KIND_FFN_GATE, KIND_FFN_UP):
        off, count = tp_window(FFN_INTERMEDIATE, degree, rank)
        return TpSlice(off, count, 0, HIDDEN)
    if ref.kind == KIND_FFN_DOWN:
        off, count = tp_window(FFN_INTERMEDIATE, degree, rank)
        return TpSlice(0, HIDDEN, off, count)
    if ref.kind == KIND_GDN_QKV:
        qk_off, qk_count = tp_window(GDN_QK_DIM, degree, rank)
        v_off, v_count = tp_window(GDN_VALUE_DIM, degree, rank)
        segments = [(qk_off, qk_count), (GDN_QK_DIM + qk_off, qk_count),
                    (2 * GDN_QK_DIM + v_off, v_count)]
        return TpFusedSlice(segments)
    if ref.kind == KIND_GDN_OUTPUT:
        off, count = tp_window(GDN_VALUE_DIM, degree, rank)
        return TpSlice(0, HIDDEN, off, count)
    if ref.kind == KIND_ATTN_QUERY:
        off, count = tp_window(2 * ATTN_Q_DIM, degree, rank)
        return TpSlice(off, count, 0, HIDDEN)
    if ref.kind in (KIND_ATTN_KEY, KIND_ATTN_VALUE):
        off, count = tp_window(ATTN_KV_DIM, degree, rank)
        return TpSlice(off, count, 0, HIDDEN)
    if ref.kind == KIND_ATTN_OUTPUT:
        off, count = tp_window(ATTN_Q_DIM, degree, rank)
        return TpSlice(0, HIDDEN, off, count)
    return None

def packed_shape(ref, plan):
    """The TP entry's rows/columns after sharding."""
    if plan is None:
        return (ref.rows, ref.columns)
    if isinstance(plan, TpFusedSlice):
        return (sum(count for _, count in plan.segments), plan.col_count)
    return (plan.row_count, plan.col_count)

def write_batch(out, source_path, offset, byte_count, mode):
    """Read byte_count payload bytes at offset. mode 0 = raw BF16,
    1 = fold +1 (standard norms), 2 = upcast BF16 to F32 (A_log/dt_bias)."""
    with open(source_path, "rb") as file:
        file.seek(offset)
        remaining = byte_count
        while remaining > 0:
            step = min(remaining, CHUNK_BYTES)
            chunk = file.read(step)
            if len(chunk) != step:
                raise PackFailure(f"short read at {offset}")
            remaining -= step
            if mode == 0:
                out.write(chunk)
            elif mode == 1:
                count = step // BF16_BYTES
                values = struct.unpack("<" + ("H" * count), chunk)
                folded = [f32_to_bf16_u16(bf16_u16_to_f32(v) + 1.0) for v in values]
                out.write(struct.pack("<" + ("H" * count), *folded))
            else:
                widened = bytearray(step * 2)
                widened[2::4] = chunk[0::2]
                widened[3::4] = chunk[1::2]
                out.write(widened)

def copy_tensor(source: SafetensorsSource, ref: TensorRef, offset: int, plan, out) -> None:
    """Stream one tensor's payload (optionally TP-sliced), upcasting BF16 to
    F32 where the pack says, folding +1 into the standard-norm weights."""
    if ref.weight_format == WEIGHT_NVFP4_PACKED:
        # The U8-packed tensor: rows unpacked, cols 2 values/byte.
        path = source.root / source.weight_map[ref.name]
        row_bytes = ref.columns // 2
        if plan is None:
            write_batch(out, path, offset, ref.rows * row_bytes, 0)
            return
        if plan.col_count == ref.columns:
            for row in range(plan.row_off, plan.row_off + plan.row_count):
                write_batch(out, path, offset + row * row_bytes, row_bytes, 0)
            return
        for row in range(plan.row_off, plan.row_off + plan.row_count):
            write_batch(out, path, offset + row * row_bytes + plan.col_off // 2,
                        plan.col_count // 2, 0)
        return
    path = source.root / source.weight_map[ref.name]
    mode = 0 if ref.weight_format == WEIGHT_FP8_E4M3_F32B128 else (
        2 if ref.weight_format == WEIGHT_F32 else (
            1 if ref.kind in NORM_PLUS_ONE_KINDS else 0))
    elem = 1 if ref.weight_format == WEIGHT_FP8_E4M3_F32B128 else BF16_BYTES
    if plan is None:
        write_batch(out, path, offset, ref.rows * ref.columns * elem, mode)
        return
    if isinstance(plan, TpFusedSlice):
        for (row_off, row_count) in plan.segments:
            for row in range(row_off, row_off + row_count):
                base = offset + (row * ref.columns) * elem
                write_batch(out, path, base, ref.columns * elem, mode)
        return
    ro = getattr(ref, "fused_up_offset", 0)
    if plan.col_count == ref.columns:
        for row in range(plan.row_off, plan.row_off + plan.row_count):
            base = offset + ((row + ro) * ref.columns) * elem
            write_batch(out, path, base, ref.columns * elem, mode)
        return
    for row in range(plan.row_off, plan.row_off + plan.row_count):
        base = offset + (((row + ro) * ref.columns) + plan.col_off) * elem
        write_batch(out, path, base, plan.col_count * elem, mode)


def copy_scale(source: SafetensorsSource, ref: TensorRef, plan, out) -> None:
    """Stream the FP8 weight_scale_inv (BF16 per 128x128 block), TP-sliced to
    mirror the weight row/col window (windows are 128-aligned)."""
    if ref.weight_format == WEIGHT_NVFP4_PACKED:
        # The e4m3 per-16 plane rows (+ the down proj's column window),
        # then the F32 weight global as the segment tail.
        path = source.root / source.weight_map[ref.scale_name]
        _, _, scale_base = source.resolve(ref.scale_name)
        plane_cols = ref.columns // 16
        g_shard, _, g_off = source.resolve(ref.nvfp4_global_name)
        if plan is None:
            write_batch(out, path, scale_base, ref.rows * plane_cols, 0)
        elif plan.col_count == ref.columns:
            for row in range(plan.row_off, plan.row_off + plan.row_count):
                write_batch(out, path, scale_base + row * plane_cols, plane_cols, 0)
        else:
            for row in range(ref.rows):
                write_batch(out, path, scale_base + row * plane_cols + plan.col_off // 16,
                            plan.col_count // 16, 0)
        with open(source.root / g_shard, "rb") as gf:
            gf.seek(g_off)
            out.write(gf.read(4))
        return
    path = source.root / source.weight_map[ref.scale_name]
    _, _, scale_base = source.resolve(ref.scale_name)
    scale_rows = ref.rows // FP8_SCALE_GROUP
    scale_cols = ref.columns // FP8_SCALE_GROUP
    if plan is None:
        write_batch(out, path, scale_base, scale_rows * scale_cols * BF16_BYTES, 2)
        return
    if isinstance(plan, TpFusedSlice):
        # The fused-tensor TP scale: each segment is a row window into the
        # SAME fused scale plane; stream its block rows (segments carry
        # 128-aligned widths by construction — GDN qk/v dims divide 128).
        for (row_off, row_count) in plan.segments:
            r0 = row_off // FP8_SCALE_GROUP
            r1 = (row_off + row_count) // FP8_SCALE_GROUP
            for row in range(r0, r1):
                base = scale_base + (row * scale_cols + 0) * BF16_BYTES
                write_batch(out, path, base, scale_cols * BF16_BYTES, 2)
        return
    ro = getattr(ref, "fused_up_offset", 0)
    r0 = (plan.row_off + ro) // FP8_SCALE_GROUP
    r1 = (plan.row_off + plan.row_count + ro) // FP8_SCALE_GROUP
    c0 = plan.col_off // FP8_SCALE_GROUP
    c1 = (plan.col_off + plan.col_count) // FP8_SCALE_GROUP
    for row in range(r0, r1):
        base = scale_base + (row * scale_cols + c0) * BF16_BYTES
        write_batch(out, path, base, (c1 - c0) * BF16_BYTES, 2)


def convert(checkpoint: Path, output: Path, first_layer: int, layer_count: int,
            receipt: dict, dry_run: bool, tp_degree: int = 1, tp_rank: int = 0,
            ffn_format: str = "bf16") -> dict:
    if ffn_format == "nvfp4a16":
        source = Nvfp4A16Source(checkpoint)
    else:
        source = SafetensorsSource(checkpoint)
    source.check_config()
    refs = build_inventory(first_layer, layer_count)
    plans = []
    cursor = 0
    for ref in refs:
        shard, meta, offset = source.check_shape(ref)
        plan = build_tp_plan(ref, tp_degree, tp_rank)
        packed_rows, packed_cols = packed_shape(ref, plan)
        payload_offset = align(cursor)
        if ref.weight_format == WEIGHT_FP8_E4M3_F32B128:
            payload_bytes = packed_rows * packed_cols * 1  # F8_E4M3
            scale_offset = align(payload_offset + payload_bytes)
            scale_bytes = (packed_rows // FP8_SCALE_GROUP) * (packed_cols // FP8_SCALE_GROUP) * F32_BYTES
        elif ref.weight_format == WEIGHT_NVFP4_PACKED:
            # U8-packed e2m1: 2 values/byte; the scale segment = the
            # per-16 e4m3 plane + the F32 weight global (tail - 4).
            payload_bytes = packed_rows * (packed_cols // 2)
            scale_offset = align(payload_offset + payload_bytes)
            scale_bytes = packed_rows * (packed_cols // 16) + 4
        else:
            element_bytes = BF16_BYTES if ref.weight_format == WEIGHT_BF16 else F32_BYTES
            payload_bytes = packed_rows * packed_cols * element_bytes
            scale_offset = 0
            scale_bytes = 0
        plans.append((ref, offset, payload_offset, payload_bytes, scale_offset, scale_bytes, plan))
        cursor = (scale_offset + scale_bytes) if scale_bytes else (payload_offset + payload_bytes)
    payload_base = align(HEADER_BYTES + len(plans) * ENTRY_BYTES)
    file_bytes = payload_base + cursor

    header = HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(plans),
        HIDDEN, layer_count, first_layer, LAYER_COUNT,
        ATTENTION_PERIOD, FULL_PHASE,
        GDN_KEY_HEADS, GDN_VALUE_HEADS, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM,
        GDN_CONV_KERNEL, ATTN_QUERY_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM,
        ATTN_ROPE_DIM, FFN_INTERMEDIATE, VOCAB, MXFP4_GROUP, MTP_LAYERS,
        tp_degree, tp_rank, HEADER_BYTES, file_bytes)
    entries = b"".join(
        ENTRY_STRUCT.pack(ref.kind, ref.layer, ref.weight_format,
                          packed_shape(ref, plan)[0], packed_shape(ref, plan)[1],
                          NVFP4_GROUP if ref.weight_format == WEIGHT_NVFP4_PACKED else (FP8_SCALE_GROUP if ref.weight_format == WEIGHT_FP8_E4M3_F32B128 else 0),
                          payload_base + payload_offset,
                          payload_bytes,
                          payload_base + scale_offset if scale_bytes else 0,
                          scale_bytes)
        for ref, _, payload_offset, payload_bytes, scale_offset, scale_bytes, plan in plans)
    receipt.update({
        "first_layer_index": first_layer,
        "layer_count": layer_count,
        "tensor_count": len(plans),
        "bytes": file_bytes,
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "source_index_sha256": source.index_sha256,
        "source_config_sha256": source.config_sha256,
    })
    if dry_run:
        print(f"qwen38_27b_stagepack slice={first_layer}+{layer_count} "
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
        for ref, source_offset, _, payload_bytes, scale_offset, scale_bytes, plan in plans:
            before = temp.tell()
            copy_tensor(source, ref, source_offset, plan, temp)
            if temp.tell() - before != payload_bytes:
                raise PackFailure(f"payload size mismatch on {ref.name}")
            if scale_bytes:
                assert temp.tell() == payload_base + scale_offset, ("scale offset drift", temp.tell(), payload_base + scale_offset)
                copy_scale(source, ref, plan, temp)
                assert temp.tell() - (payload_base + scale_offset) == scale_bytes, ("scale size mismatch", ref.name)
            pad = align(temp.tell()) - temp.tell()
            if pad:
                temp.write(b"\0" * pad)
        temp.flush()
        os.fsync(temp.fileno())
    os.replace(temp_path, output)
    receipt["output_sha256"] = sha256_file(output)
    receipt["file"] = str(output)
    print(f"qwen38_27b_stagepack slice={first_layer}+{layer_count} tensors={len(plans)} "
          f"file_gib={file_bytes / 2**30:.2f} wrote {output}")
    return receipt


def verify(pack_path: Path) -> dict:
    """Parse a pack back and check every rule the format header states."""
    file_bytes = pack_path.stat().st_size
    with pack_path.open("rb") as file:
        raw_header = file.read(HEADER_BYTES)
        if len(raw_header) != HEADER_BYTES:
            raise PackFailure("short header")
        fields = HEADER_STRUCT.unpack(raw_header)
        (magic, version, header_bytes, entry_bytes, tensor_count, hidden,
         layer_count, first_layer, total_layers, period, phase,
         gdn_kh, gdn_vh, gdn_kd, gdn_vd, conv_kernel,
         attn_qh, attn_kvh, attn_hd, rope_dim, ffn, vocab, mxfp4, mtp,
         tp_degree, tp_rank, directory_offset, declared_bytes) = fields
        geometry = {
            "magic": (magic, MAGIC), "format_version": (version, FORMAT_VERSION),
            "header_bytes": (header_bytes, HEADER_BYTES),
            "directory_entry_bytes": (entry_bytes, ENTRY_BYTES),
            "tensor_count": (tensor_count, expected_tensor_count(first_layer, layer_count)),
            "hidden_dimension": (hidden, HIDDEN), "layer_count": (layer_count, layer_count),
            "total_layer_count": (total_layers, LAYER_COUNT),
            "attention_period": (period, ATTENTION_PERIOD),
            "full_attention_phase": (phase, FULL_PHASE),
            "gdn_key_head_count": (gdn_kh, GDN_KEY_HEADS),
            "gdn_value_head_count": (gdn_vh, GDN_VALUE_HEADS),
            "gdn_head_key_dimension": (gdn_kd, GDN_HEAD_KEY_DIM),
            "gdn_head_value_dimension": (gdn_vd, GDN_HEAD_VALUE_DIM),
            "gdn_conv_kernel": (conv_kernel, GDN_CONV_KERNEL),
            "attn_query_head_count": (attn_qh, ATTN_QUERY_HEADS),
            "attn_kv_head_count": (attn_kvh, ATTN_KV_HEADS),
            "attn_head_dimension": (attn_hd, ATTN_HEAD_DIM),
            "attn_rope_dimension": (rope_dim, ATTN_ROPE_DIM),
            "ffn_intermediate_dimension": (ffn, FFN_INTERMEDIATE),
            "output_vocab_count": (vocab, VOCAB),
            "mxfp4_group_size": (mxfp4, MXFP4_GROUP), "mtp_layer_count": (mtp, MTP_LAYERS),
            "tp_degree": (tp_degree, tp_degree), "tp_rank": (tp_rank, tp_rank),
        }
        for name, (actual, expected) in geometry.items():
            if actual != expected:
                raise PackFailure(f"geometry field {name}: {actual}, expected {expected}")
        if directory_offset != HEADER_BYTES or declared_bytes != file_bytes:
            raise PackFailure("directory offset or file size mismatch")
        raw_directory = file.read(tensor_count * ENTRY_BYTES)
        if len(raw_directory) != tensor_count * ENTRY_BYTES:
            raise PackFailure("short directory")
        payload_base = align(HEADER_BYTES + tensor_count * ENTRY_BYTES)
        cursor = payload_base
        seen = set()
        for index in range(tensor_count):
            entry = ENTRY_STRUCT.unpack_from(raw_directory, index * ENTRY_BYTES)
            (kind, layer, weight_format, rows, columns, scale_group,
             payload_offset, payload_bytes, scale_offset, scale_bytes) = entry
            expected_rows, expected_columns, expected_format = kind_shape(kind)
            is_fp8 = weight_format == WEIGHT_FP8_E4M3_F32B128
            is_nvfp4 = weight_format == WEIGHT_NVFP4_PACKED
            if tp_degree > 1:
                class _Ref:
                    pass
                _ref = _Ref()
                _ref.kind, _ref.layer = kind, layer
                _ref.rows, _ref.columns = expected_rows, expected_columns
                plan = build_tp_plan(_ref, tp_degree, tp_rank)
                expected_rows, expected_columns = packed_shape(_ref, plan)
            if (rows, columns) != (expected_rows, expected_columns):
                raise PackFailure(f"entry {index} kind {kind}: shape mismatch")
            if kind in FP8_KINDS:
                if weight_format not in (expected_format, WEIGHT_FP8_E4M3_F32B128, WEIGHT_NVFP4_PACKED):
                    raise PackFailure(f"entry {index} kind {kind}: format mismatch")
            elif weight_format != expected_format:
                raise PackFailure(f"entry {index} kind {kind}: format mismatch")
            if (kind, layer) in seen:
                raise PackFailure(f"duplicate tensor kind {kind} layer {layer}")
            seen.add((kind, layer))
            if layer == GLOBAL_LAYER:
                if kind not in GLOBAL_TENSORS:
                    raise PackFailure(f"entry {index}: per-layer kind {kind} at the global marker")
            elif layer == MTP_LAYER:
                if kind not in EVERY_LAYER_KINDS + ATTN_LAYER_KINDS:
                    raise PackFailure(f"entry {index}: kind {kind} not valid at the MTP marker")
            else:
                if layer >= LAYER_COUNT:
                    raise PackFailure(f"entry {index}: layer {layer} out of range")
                if kind in GDN_LAYER_KINDS and not is_gdn_layer(layer):
                    raise PackFailure(f"entry {index}: GDN kind {kind} on attention layer {layer}")
                if kind in ATTN_LAYER_KINDS and is_gdn_layer(layer):
                    raise PackFailure(f"entry {index}: attention kind {kind} on GDN layer {layer}")
            if payload_offset != align(cursor) or payload_offset % PAYLOAD_ALIGNMENT != 0:
                raise PackFailure(f"entry {index}: payload offset {payload_offset}, expected {align(cursor)}")
            if is_fp8:
                expected_scale_bytes = (rows // FP8_SCALE_GROUP) * (columns // FP8_SCALE_GROUP) * F32_BYTES
                if scale_group != FP8_SCALE_GROUP or scale_offset == 0 or scale_bytes != expected_scale_bytes:
                    raise PackFailure(f"entry {index}: FP8 scale metadata mismatch")
                element_bytes = 1
            elif is_nvfp4:
                if scale_group != NVFP4_GROUP or scale_offset == 0:
                    raise PackFailure(f"entry {index}: nvfp4 scale metadata mismatch")
                expected_scale_bytes = rows * (columns // 16) + 4
                if scale_bytes != expected_scale_bytes:
                    raise PackFailure(f"entry {index}: nvfp4 scale byte count mismatch")
                element_bytes = 0  # packed: counted as rows * (columns // 2) below
            else:
                element_bytes = BF16_BYTES if weight_format == WEIGHT_BF16 else F32_BYTES
                if scale_group != 0 or scale_offset != 0 or scale_bytes != 0:
                    raise PackFailure(f"entry {index}: BF16/F32 tensors carry no scales")
            expected_payload = (rows * (columns // 2) if is_nvfp4
                                else rows * columns * element_bytes)
            if payload_bytes != expected_payload:
                raise PackFailure(f"entry {index}: payload byte count mismatch")
            cursor = (scale_offset + scale_bytes) if is_fp8 else (payload_offset + payload_bytes)
        if align(cursor) != align(file_bytes):
            raise PackFailure("trailing payload does not close the file")
    return {"file": str(pack_path), "bytes": file_bytes, "tensor_count": tensor_count,
            "first_layer_index": first_layer, "layer_count": layer_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint directory")
    parser.add_argument("--output", type=Path, help="pack output path")
    parser.add_argument("--stage-index", type=int, help="stage index into --recipe")
    parser.add_argument("--recipe", type=Path, help="PP recipe JSON (layer split + receipt hash)")
    parser.add_argument("--first-layer", type=int, help="explicit slice start")
    parser.add_argument("--layer-count", type=int, help="explicit slice length")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--receipt", type=Path, help="receipt output (default: <output>.receipt.json)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", type=Path, help="verify an existing pack and exit")
    parser.add_argument("--tp-degree", type=int, default=1,
                        help="tensor-parallel degree; shards the pack for --tp-rank")
    parser.add_argument("--tp-rank", type=int, default=0,
                        help="this rank's shard (0 .. tp-degree-1)")
    parser.add_argument("--ffn-format", choices=("bf16", "nvfp4a16"), default="bf16",
                        help="nvfp4a16 = the main-layer FFN projections from the "
                             "nvfp4a16-bf16-spine release on the NVFP4 wire "
                             "(weight-only, BF16 activations); default bf16")
    args = parser.parse_args()
    if args.tp_degree < 1 or args.tp_rank < 0 or args.tp_rank >= args.tp_degree:
        parser.error("--tp-rank must satisfy 0 <= tp-rank < tp-degree")
    if args.tp_degree > 1 and (args.first_layer != 0 or args.layer_count != LAYER_COUNT):
        # Combined TP+PP mode: the inventory is stage-aware (embedding on
        # stage zero, head + MTP on the head stage) and build_tp_plan
        # shards per tensor, so the two dimensions compose. The module
        # side must load stage-sliced TP packs via the TP4PP4 rank map.
        print(f"qwen38_27b_stagepack: combined TP{args.tp_degree}+PP slice "
              f"{args.first_layer}+{args.layer_count} (module needs the "
              f"TP4PP4 rank map; single-pack full-stack loaders will reject it)")

    if args.verify is not None:
        result = verify(args.verify)
        print(f"qwen38_27b_stagepack verify ok: {result['file']} "
              f"slice={result['first_layer_index']}+{result['layer_count']} "
              f"tensors={result['tensor_count']} bytes={result['bytes']}")
        return 0

    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    if args.stage_index is not None:
        if args.recipe is None:
            parser.error("--stage-index requires --recipe")
        recipe = json.loads(args.recipe.read_text())
        stage = recipe["pp"]["stages"][args.stage_index]
        first_layer, layer_count = stage["first_layer_index"], stage["layer_count"]
    elif args.first_layer is not None and args.layer_count is not None:
        first_layer, layer_count = args.first_layer, args.layer_count
    else:
        parser.error("name --stage-index with --recipe, or --first-layer with --layer-count")
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run")

    receipt = {
        "kind": "sparkpipe.qwen38_27b.stagepack-receipt.v1",
        "tool": "tools/qwen38_27b_stagepack.py",
        "ffn_format": args.ffn_format,
        "checkpoint": str(args.checkpoint),
        "contract": {"path": str(args.contract),
                     "sha256": sha256_file(args.contract) if args.contract.is_file() else None},
        "recipe": None,
        "stage_index": args.stage_index,
        "weight_formats": {"ffn": args.ffn_format, "projections": "bf16", "gdn_a_log_dt_bias": "f32"},
    }
    if args.recipe is not None:
        receipt["recipe"] = {"path": str(args.recipe),
                             "sha256": sha256_file(args.recipe),
                             "content_hash": recipe.get("content_hash")}
    result = convert(args.checkpoint, args.output or Path("/dev/null"),
                     first_layer, layer_count, receipt, args.dry_run,
                     args.tp_degree, args.tp_rank,
                     ffn_format=args.ffn_format)
    if not args.dry_run:
        receipt_path = args.receipt or Path(str(args.output) + ".receipt.json")
        write_receipt(result, receipt_path, suffix=None)
        print(f"qwen38_27b_stagepack receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen38_27b_stagepack: {error}", file=sys.stderr)
        sys.exit(1)
