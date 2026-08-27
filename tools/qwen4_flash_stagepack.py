#!/usr/bin/env python3
"""Convert the Qwen/Qwen3.8-Flash-Next BF16 safetensors checkpoint into
qwen4_flash stage packs (lane qwen-flash, milestone M4).

Setup-time code, never the serving path. Mirrors tools/qwen38_stagepack.py
and tools/spark_pack_common.py for the linear-attention/attention/MoE
inventory, re-parameterized for the Flash geometry, and adds:

  * BF16 -> FP8_E4M3 quantization of the routed experts at pack time (the
    Flash checkpoint ships BF16 unquantized; the module's expert codec is
    block-128 scaled FP8). Default scale plane is F32 (format 4, the codec
    the module validates today); --expert-format fp8-e8m0b128 emits the
    format-6 MX plane (E4M3 payload + E8M0 scale bytes per 128 block, the
    qwen38_27b serving format) for the module format bump.
  * --tp-degree/--tp-rank rank-local packs with the family TP plan: fused
    qkv/conv triple-sliced by head groups, projections column/row-parallel,
    experts expert-sharded, norms/globals replicated, the MTP decoder
    slicing like main layers per the family rule.

Checkpoint layout pinned against the warm-copy index at revision
f5d0827 (model_contracts/qwen4_flash_authoritative.json):

  * linear-attn layers (36): linear_attn.{in_proj_qkv [10240,2560] fused
    q2048|k2048|v6144, in_proj_z [6144,2560], in_proj_a/in_proj_b [48,2560],
    conv1d [10240,1,4], A_log [48], dt_bias [48], norm [128],
    out_proj [2560,6144]}.
  * full-attention layers (12): self_attn.{q_proj [12288,2560] fused
    query|sigmoid gate, k_proj/v_proj [512,2560], o_proj [2560,6144],
    q_norm/k_norm [256]}.
  * every layer: mlp.gate.weight [512,2560], mlp.experts.gate_up_proj
    [512,1280,2560] (fused w1|w3, split here), mlp.experts.down_proj
    [512,2560,640], mlp.shared_expert.{gate_proj,up_proj [640,2560],
    down_proj [2560,640]}, mlp.shared_expert_gate.weight [2560].
  * head: lm_head [248320,2560], embed [248320,2560]. NOTE: the Flash
    final readout is the hyper_connection_mixer, not a plain model.norm;
    the FINAL_NORM slot is filled from the mixer hc_norm stream-0 section
    (receipted approximation, see hc_approximations).
  * MTP: mtp.fc_embedding [2560,2560] + mtp.fc_hidden [2560,2560] packed
    column-concatenated into the [2560,5120] MTP_FC slot; pre_fc_norm_* as
    the three norm slots (hidden norm from the 10240-wide tensor's
    stream-0 section); one full-attention decoder layer at the MTP marker.

The vision tower (model.visual.*), the per-sublayer hyper-connection
mixers, the attention indexer, and the layer-1 PLE block are OUT OF SCOPE
for the module contract this revision; they are enumerated in the receipt
under unmapped_checkpoint_tensors and carried as integration requests.
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
    tp_shard_range,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "model_contracts" / "qwen4_flash_authoritative.json"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

# Wire constants, mirroring spark_qwen4_flash_stagepack_format.h.
MAGIC = 0x50533451  # 'Q4SP' little endian
FORMAT_VERSION = 1
HEADER_BYTES = 120
ENTRY_BYTES = 56
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE
PAYLOAD_ALIGNMENT = 256

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_F32B128 = 4
WEIGHT_FP8_E8M0B128 = 6

HIDDEN = 2560
LAYER_COUNT = 48
ATTENTION_PERIOD = 4
FULL_PHASE = 3
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_HEAD_KEY_DIM = 128
GDN_HEAD_VALUE_DIM = 128
GDN_CONV_KERNEL = 4
ATTN_QUERY_HEADS = 24
ATTN_KV_HEADS = 2
ATTN_HEAD_DIM = 256
ATTN_ROPE_DIM = 64
EXPERT_COUNT = 512
EXPERTS_PER_TOKEN = 10
EXPERT_INTERMEDIATE = 640
VOCAB = 248320
MTP_LAYERS = 1
MXFP4_GROUP = 32
FP8_BLOCK = 128

GDN_QK_DIM = GDN_KEY_HEADS * GDN_HEAD_KEY_DIM            # 2048
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_VALUE_DIM     # 6144
GDN_CONV_CHANNELS = 2 * GDN_QK_DIM + GDN_VALUE_DIM       # 10240
ATTN_Q_DIM = ATTN_QUERY_HEADS * ATTN_HEAD_DIM            # 6144
ATTN_KV_DIM = ATTN_KV_HEADS * ATTN_HEAD_DIM              # 512

HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES

# Tensor kinds, mirroring SparkQwen4FlashStagePackTensorKind.
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
LAYER_PREFIX = "model.language_model.layers."

# E4M3 representable max (finite) for amax scaling.
FP8_E4M3_MAX = 448.0


def is_gdn_layer(layer_index: int) -> bool:
    return (layer_index % ATTENTION_PERIOD) != FULL_PHASE


def kind_shape(kind: int) -> tuple[int, int, int]:
    table = {
        KIND_EMBEDDING: (VOCAB, HIDDEN, WEIGHT_BF16),
        KIND_FINAL_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_LM_HEAD: (VOCAB, HIDDEN, WEIGHT_BF16),
        KIND_ATTENTION_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MLP_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MOE_GATE: (EXPERT_COUNT, HIDDEN, WEIGHT_BF16),
        KIND_MOE_W1: (EXPERT_COUNT * EXPERT_INTERMEDIATE, HIDDEN, WEIGHT_FP8_F32B128),
        KIND_MOE_W3: (EXPERT_COUNT * EXPERT_INTERMEDIATE, HIDDEN, WEIGHT_FP8_F32B128),
        KIND_MOE_DOWN: (EXPERT_COUNT * HIDDEN, EXPERT_INTERMEDIATE, WEIGHT_FP8_F32B128),
        KIND_MOE_SHARED_GATE: (EXPERT_INTERMEDIATE, HIDDEN, WEIGHT_BF16),
        KIND_MOE_SHARED_UP: (EXPERT_INTERMEDIATE, HIDDEN, WEIGHT_BF16),
        KIND_MOE_SHARED_DOWN: (HIDDEN, EXPERT_INTERMEDIATE, WEIGHT_BF16),
        KIND_MOE_SHARED_GATE_WEIGHT: (1, HIDDEN, WEIGHT_BF16),
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


def layer_tensor_name(kind: int, layer: int) -> str:
    prefix = f"{LAYER_PREFIX}{layer}." if layer != MTP_LAYER else MTP_PREFIX + "layers.0."
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
        # Flash has no input/post_attention_layernorm: the hyper-connection
        # hc_norm carries the per-sublayer norm role over the 4-stream
        # residual; the stream-0 section fills the [1, H] slot (receipted
        # under hc_approximations).
        KIND_ATTENTION_NORM: "attn_hyper_connection.hc_norm.weight",
        KIND_MLP_NORM: "mlp_hyper_connection.hc_norm.weight",
        KIND_MOE_GATE: "mlp.gate.weight",
        KIND_MOE_W1: "mlp.experts.gate_up_proj",
        KIND_MOE_W3: "mlp.experts.gate_up_proj",
        KIND_MOE_DOWN: "mlp.experts.down_proj",
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
    KIND_EMBEDDING: "model.language_model.embed_tokens.weight",
    # Flash readout: no plain model.norm; stream-0 section of the mixer's
    # hc_norm stands in (receipted under hc_approximations).
    KIND_FINAL_NORM: "model.language_model.hyper_connection_mixer.hc_norm.weight",
    KIND_LM_HEAD: "lm_head.weight",
    KIND_MTP_FC: MTP_PREFIX + "fc_embedding.weight+" + MTP_PREFIX + "fc_hidden.weight",
    KIND_MTP_EMBED_NORM: MTP_PREFIX + "pre_fc_norm_embedding.weight",
    KIND_MTP_HIDDEN_NORM: MTP_PREFIX + "pre_fc_norm_hidden.weight",
    KIND_MTP_FINAL_NORM: MTP_PREFIX + "pre_fc_norm_embedding.weight",
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

# Replicated (never sliced) kinds: norms, scalars, and the MTP globals per
# the family rule in spark_pack_common.spark_pack_replicated_draft_rows.
# KV projections replicate whenever kv_heads < tp_degree (Flash has 2 kv
# heads; TP4 replicates them - receipted under tp_replicated_kv).
REPLICATED_KINDS = frozenset({
    KIND_ATTENTION_NORM, KIND_MLP_NORM, KIND_MOE_SHARED_GATE_WEIGHT,
    KIND_GDN_NORM, KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM,
    KIND_FINAL_NORM, KIND_MTP_FC, KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM,
    KIND_MTP_FINAL_NORM,
    # The router gate MUST be replicated: a rank-local gate would select a
    # rank-local top-k, and the union across ranks is not the model's top-k.
    # Every rank scores all experts and routes the same global top-k; each
    # rank executes only the pairs whose experts live in its shard (the
    # module derives the route group base from the gate width). The first
    # deployed pack generation narrowed it - the module still accepts and
    # runs those (self-consistent, wrong mixture); rebuilt packs replicate.
    KIND_MOE_GATE,
})


class TensorRef:
    def __init__(self, kind: int, layer: int, name: str):
        self.kind = kind
        self.layer = layer
        self.name = name
        self.rows, self.columns, self.weight_format = kind_shape(kind)
        # Fused gate_up_proj [E, 2*I, H] splits into W1 (rows 0:I of each
        # expert's 1280) and W3 (rows I:2I); down_proj [E, H, I] flattens
        # expert-major with no slice.
        if kind in (KIND_MOE_W1, KIND_MOE_W3):
            self.slice_start = 0 if kind == KIND_MOE_W1 else EXPERT_INTERMEDIATE
            self.slice_rows = EXPERT_INTERMEDIATE
        else:
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


def build_inventory(first_layer: int, layer_count: int) -> list[TensorRef]:
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
        if first_layer != 0:
            refs.append(TensorRef(KIND_EMBEDDING, GLOBAL_LAYER, GLOBAL_TENSORS[KIND_EMBEDDING]))
        for kind in (KIND_FINAL_NORM, KIND_LM_HEAD, KIND_MTP_FC,
                     KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM):
            refs.append(TensorRef(kind, GLOBAL_LAYER, GLOBAL_TENSORS[kind]))
        for kind in EVERY_LAYER_KINDS + ATTN_LAYER_KINDS:
            refs.append(TensorRef(kind, MTP_LAYER, layer_tensor_name(kind, MTP_LAYER))
)
    expected = expected_tensor_count(first_layer, layer_count)
    if len(refs) != expected:
        raise PackFailure(f"inventory {len(refs)} tensors, format expects {expected}")
    return refs


class SafetensorsSource(_BaseSafetensorsSource):
    """qwen4_flash checkpoint reader: shared index/header/payload resolution
    plus this model's config expectations."""

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
            "shared_expert_intermediate_size": EXPERT_INTERMEDIATE,
            "vocab_size": VOCAB,
            "full_attention_interval": ATTENTION_PERIOD,
            "tie_word_embeddings": False,
        }
        super().check_config(expectations, section="text_config")

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        name, rows, columns = ref.name, ref.rows, ref.columns
        if ref.kind == KIND_MTP_FC:
            # [H,2H] column-concat of fc_embedding and fc_hidden.
            for part, expect_rows in ((name.split("+")[0], rows), (name.split("+")[1], HIDDEN)):
                shard, meta, offset = self.resolve(part)
                if meta["dtype"] != "BF16" or meta["shape"] != [expect_rows, HIDDEN]:
                    raise PackFailure(f"{part}: {meta['dtype']} {meta['shape']}, expected BF16 [{expect_rows},{HIDDEN}]")
                del shard
            return self.resolve(name.split("+")[0])
        if ref.kind == KIND_GDN_CONV_WEIGHT:
            shard, meta, offset = self.resolve(name)
            if meta["dtype"] != "BF16" or meta["shape"] != [rows, 1, columns]:
                raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected BF16 [{rows},1,{columns}]")
            return shard, meta, offset
        if ref.kind in (KIND_FINAL_NORM, KIND_MTP_HIDDEN_NORM, KIND_ATTENTION_NORM, KIND_MLP_NORM):
            shard, meta, offset = self.resolve(name)
            if ref.kind in (KIND_ATTENTION_NORM, KIND_MLP_NORM) and ref.layer != GLOBAL_LAYER:
                prefix = f"{LAYER_PREFIX}{ref.layer}." if ref.layer != MTP_LAYER else MTP_PREFIX + "layers.0."
                shard, meta, offset = self.resolve(prefix + ("attn_hyper_connection.hc_norm.weight" if ref.kind == KIND_ATTENTION_NORM else "mlp_hyper_connection.hc_norm.weight"))
            if meta["dtype"] != "BF16" or meta["shape"] != [4 * HIDDEN]:
                raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected BF16 [{4 * HIDDEN}] (4-stream hc norm)")
            return shard, meta, offset
        if ref.kind in (KIND_GDN_A_LOG, KIND_GDN_DT_BIAS):
            shard, meta, offset = self.resolve(name)
            if meta["dtype"] != "BF16" or meta["shape"] != [columns]:
                raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected BF16 [{columns}]")
            return shard, meta, offset
        if ref.kind in (KIND_MOE_W1, KIND_MOE_W3):
            shard, meta, offset = self.resolve(name)
            if meta["dtype"] != "BF16" or meta["shape"] != [EXPERT_COUNT, 2 * EXPERT_INTERMEDIATE, HIDDEN]:
                raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected BF16 [{EXPERT_COUNT},{2 * EXPERT_INTERMEDIATE},{HIDDEN}]")
            return shard, meta, offset
        if ref.kind == KIND_MOE_DOWN:
            shard, meta, offset = self.resolve(name)
            if meta["dtype"] != "BF16" or meta["shape"] != [EXPERT_COUNT, HIDDEN, EXPERT_INTERMEDIATE]:
                raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected BF16 [{EXPERT_COUNT},{HIDDEN},{EXPERT_INTERMEDIATE}]")
            return shard, meta, offset
        return super().check_shape(name, rows, columns)


# -- BF16 -> FP8_E4M3 block quantization ---------------------------------------


def quantize_fp8_blocks(matrix_bf16: "np.ndarray", expert_format: str):
    """Quantize one [R, C] BF16 matrix to FP8_E4M3. Returns (payload bytes,
    scale bytes). F32B128: per-128x128-tile f32 multiplier scales. E8M0B128:
    one exponent byte per (row, 128-column block) - the per-row MX layout the
    module's grouped expert kernels decode (SparkLmDotRowFp8E8m0: scale index
    = row * (columns/128) + block). E8M0 scales round UP to a power of two so
    value/scale never exceeds the E4M3 max (no saturation).

    The Flash checkpoint ships BF16, so this is a real quantization step
    (unlike the vendor-FP8 qwen38_max source): amax per block, scale =
    amax / 448, payload = round-to-nearest-even code * scale (values are
    divided by the scale; the module's scale plane is a multiplier).
    """
    import numpy as np
    rows, columns = matrix_bf16.shape
    if rows % FP8_BLOCK != 0 or columns % FP8_BLOCK != 0:
        raise PackFailure(f"fp8 block quantization needs {FP8_BLOCK}-aligned matrix, got [{rows},{columns}]")
    if expert_format == "fp8-f32b128":
        values = matrix_bf16.astype(np.float32).reshape(
            rows // FP8_BLOCK, FP8_BLOCK, columns // FP8_BLOCK, FP8_BLOCK)
        amax = np.maximum(np.abs(values).max(axis=(1, 3), keepdims=True), 1e-30)
        scale = amax / FP8_E4M3_MAX
        scaled = values / scale
        payload = np.clip(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
        codes = float_to_e4m3(payload)
        payload_bytes = codes.astype(np.uint8).tobytes()
        scale_plane = scale.astype("<f4").tobytes()
        return payload_bytes, scale_plane
    if expert_format == "fp8-e8m0b128":
        values = matrix_bf16.astype(np.float32).reshape(rows, columns // FP8_BLOCK, FP8_BLOCK)
        amax = np.maximum(np.abs(values).max(axis=2, keepdims=True), 1e-30)
        scale = amax / FP8_E4M3_MAX
        # ceil to a power of two: scale_rounded >= scale keeps |v|/scale <= 448.
        exponent = np.minimum(np.maximum(np.ceil(np.log2(scale)), -127.0), 127.0)
        scale_rounded = np.power(2.0, exponent)
        scaled = values / scale_rounded
        payload = np.clip(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
        codes = float_to_e4m3(payload)
        payload_bytes = codes.astype(np.uint8).reshape(rows, columns).tobytes()
        scale_plane = (exponent + 127.0).astype(np.uint8).reshape(rows, columns // FP8_BLOCK).tobytes()
        return payload_bytes, scale_plane
    raise PackFailure(f"unknown expert format {expert_format}")


def float_to_e4m3(values: "np.ndarray") -> "np.ndarray":
    """Round-to-nearest-even conversion to E4M3 codes (no inf; values beyond
    +/-448 saturate to code 0x7e; zero maps to 0; NaN maps to 0x7f)."""
    import numpy as np
    f32 = values.astype(np.float32)
    sign = np.signbit(f32).astype(np.uint8)
    magnitude = np.abs(f32)
    codes = np.zeros(magnitude.shape, dtype=np.uint8)
    zero = magnitude == 0.0
    # Normals: value = (1 + m/8) * 2^e with e = floor(log2(magnitude)) so the
    # fraction magnitude / 2^e lands in [1, 2); round the 3-bit fraction RNE.
    true_exponent = np.floor(np.log2(np.where(zero, 1.0, magnitude))).astype(np.int32)
    # Subnormal range: magnitude < 2^-6 encodes as (m/8) * 2^-6 with an
    # integer mantissa m = round(magnitude / 2^-9); magnitudes below half
    # the smallest subnormal step round to zero.
    sub = true_exponent < -6
    exponent = np.where(sub, -6, true_exponent)
    power = np.power(2.0, exponent.astype(np.float32))
    fraction = np.where(zero, 0.0, magnitude / power)
    rounded = np.round(fraction * 8.0) / 8.0
    carry = rounded >= 2.0
    rounded = np.where(carry, 1.0, rounded)
    exponent = np.where(carry, exponent + 1, exponent)
    mantissa = np.clip(np.round((rounded - 1.0) * 8.0), 0, 7).astype(np.int32)
    field = np.where(sub, 0, np.clip(exponent + 7, 1, 15)).astype(np.int32)
    sub_mantissa = np.clip(np.round(magnitude / (2.0 ** -9)), 0, 7).astype(np.int32)
    final_mantissa = np.where(sub, sub_mantissa, mantissa)
    codes = (sign << 7) | (field.astype(np.uint8) << 3) | final_mantissa.astype(np.uint8)
    codes = np.where(zero, np.uint8(0), codes)
    # Saturate overflow to the max finite magnitude +/-448 (code 0x7e).
    overflow = magnitude > FP8_E4M3_MAX
    codes = np.where(overflow, (sign << 7) | np.uint8(0x7E), codes)
    return codes


# -- TP sharding ----------------------------------------------------------------


def shard_ref(ref: TensorRef, tp_degree: int, tp_rank: int) -> TensorRef:
    """Return this rank's view of a tensor ref: the ref's rows/columns are
    narrowed in place and the slice metadata recorded for the copier.

    Plan (family-standard):
      * replicated kinds and all [1, C] norms: whole.
      * attention: q rows fused head-blocks (query then gate per head, so
        the fused pair stays together); k/v rows by kv-head blocks; o
        columns by query-head blocks.
      * linear-attn: qkv rows and conv rows triple-sliced by head groups
        (q plane K/tp heads, k plane K/tp heads, v plane V/tp heads,
        concatenated in the q|k|v order); gate rows by value-head blocks;
        beta/decay rows by value heads; out columns by value-head blocks;
        A_log/dt_bias columns by value heads.
      * MoE: gate rows by expert blocks; w1/w3/down rows by expert blocks
        (FP8 scale rows stay block-aligned: 128 experts = 81920 rows);
        shared expert gate/up rows and down columns by intermediate blocks.
      * embedding/lm head rows by vocab blocks.
      * MTP layer kinds slice like main layers (family rule).
    """
    if tp_degree == 1 or ref.kind in REPLICATED_KINDS:
        ref.tp_degree, ref.tp_rank = 1, 0
        return ref
    if ref.kind in (KIND_ATTN_KEY, KIND_ATTN_VALUE) and ATTN_KV_HEADS % tp_degree != 0:
        ref.tp_degree, ref.tp_rank = 1, 0
        return ref

    if ref.kind == KIND_ATTN_QUERY:
        start, count = tp_shard_range(ATTN_QUERY_HEADS, tp_degree, tp_rank)
        ref.rows = count * 2 * ATTN_HEAD_DIM
        ref.row_slice = (start * 2 * ATTN_HEAD_DIM, ref.rows)
    elif ref.kind in (KIND_ATTN_KEY, KIND_ATTN_VALUE):
        start, count = tp_shard_range(ATTN_KV_HEADS, tp_degree, tp_rank)
        ref.rows = count * ATTN_HEAD_DIM
        ref.row_slice = (start * ATTN_HEAD_DIM, ref.rows)
    elif ref.kind == KIND_ATTN_OUTPUT:
        start, count = tp_shard_range(ATTN_Q_DIM, tp_degree, tp_rank, block=FP8_BLOCK)
        ref.columns = count
        ref.column_slice = (start, count)
    elif ref.kind in (KIND_GDN_QKV, KIND_GDN_CONV_WEIGHT):
        key_start, key_count = tp_shard_range(GDN_KEY_HEADS, tp_degree, tp_rank)
        value_start, value_count = tp_shard_range(GDN_VALUE_HEADS, tp_degree, tp_rank)
        ref.triple_slice = (
            key_start * GDN_HEAD_KEY_DIM, key_count * GDN_HEAD_KEY_DIM,
            value_start * GDN_HEAD_VALUE_DIM, value_count * GDN_HEAD_VALUE_DIM)
        width = key_count * GDN_HEAD_KEY_DIM
        ref.rows = 2 * width + value_count * GDN_HEAD_VALUE_DIM
    elif ref.kind == KIND_GDN_GATE:
        start, count = tp_shard_range(GDN_VALUE_DIM, tp_degree, tp_rank, block=FP8_BLOCK)
        ref.rows = count
        ref.row_slice = (start, count)
    elif ref.kind in (KIND_GDN_BETA, KIND_GDN_DECAY, KIND_GDN_A_LOG, KIND_GDN_DT_BIAS):
        start, count = tp_shard_range(GDN_VALUE_HEADS, tp_degree, tp_rank)
        if ref.kind in (KIND_GDN_BETA, KIND_GDN_DECAY):
            ref.rows = count
        else:
            ref.columns = count
        ref.row_slice = (start, count)
    elif ref.kind == KIND_GDN_OUTPUT:
        start, count = tp_shard_range(GDN_VALUE_DIM, tp_degree, tp_rank, block=FP8_BLOCK)
        ref.columns = count
        ref.column_slice = (start, count)
    elif ref.kind == KIND_MOE_GATE:
        start, count = tp_shard_range(EXPERT_COUNT, tp_degree, tp_rank)
        ref.rows = count
        ref.row_slice = (start, count)
    elif ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN):
        experts_per_rank = EXPERT_COUNT // tp_degree
        start = tp_rank * experts_per_rank
        ref.expert_slice = (start, experts_per_rank)
        rows_per_expert = ref.rows // EXPERT_COUNT
        ref.rows = experts_per_rank * rows_per_expert
    elif ref.kind in (KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_UP):
        start, count = tp_shard_range(EXPERT_INTERMEDIATE, tp_degree, tp_rank)
        ref.rows = count
        ref.row_slice = (start, count)
    elif ref.kind == KIND_MOE_SHARED_DOWN:
        start, count = tp_shard_range(EXPERT_INTERMEDIATE, tp_degree, tp_rank)
        ref.columns = count
        ref.column_slice = (start, count)
    elif ref.kind in (KIND_EMBEDDING, KIND_LM_HEAD):
        start, count = tp_shard_range(VOCAB, tp_degree, tp_rank)
        ref.rows = count
        ref.row_slice = (start, count)
    else:
        raise PackFailure(f"kind {ref.kind} has no TP shard rule")
    ref.tp_degree, ref.tp_rank = tp_degree, tp_rank
    return ref


# -- payload writers ------------------------------------------------------------


def read_source_matrix(source: SafetensorsSource, name: str) -> "np.ndarray":
    """Read a 2-D BF16 tensor from the checkpoint as a numpy u16 view."""
    import numpy as np
    shard, meta, offset = source.resolve(name)
    shape = meta["shape"]
    elements = 1
    for extent in shape:
        elements *= extent
    with (source.root / shard).open("rb") as file:
        file.seek(offset)
        raw = file.read(elements * BF16_BYTES)
    if len(raw) != elements * BF16_BYTES:
        raise PackFailure(f"short read on {name}")
    packed = np.frombuffer(raw, dtype="<u2")
    if len(shape) == 3:
        packed = packed.reshape(shape)
    elif len(shape) == 2:
        packed = packed.reshape(shape)
    elif len(shape) == 1:
        packed = packed.reshape(shape[0])
    else:
        packed = packed.reshape(shape)
    return packed


def bf16_widen(packed_u16) -> "np.ndarray":
    """u16 bf16 bits -> f32 values (bit shift, no value conversion)."""
    import numpy as np
    bits = packed_u16.astype(np.uint32) << 16
    return bits.view(np.float32)


def bf16_to_f32_matrix(packed_u16) -> "np.ndarray":
    import numpy as np
    bits = packed_u16.astype(np.uint32) << 16
    return bits.view(np.float32)


def copy_sharded_bf16(source: SafetensorsSource, ref: TensorRef, offset: int, out) -> None:
    import numpy as np
    if getattr(ref, "triple_slice", None):
        # Fused q|k|v planes re-gathered per rank.
        key_start, key_count, value_start, value_count = ref.triple_slice
        full = read_source_matrix(source, ref.name)
        if full.ndim == 3:  # conv1d [C,1,K]
            full = full[:, 0, :]
        q = full[key_start:key_start + key_count, :]
        k = full[GDN_QK_DIM + key_start:GDN_QK_DIM + key_start + key_count, :]
        v = full[2 * GDN_QK_DIM + value_start:2 * GDN_QK_DIM + value_start + value_count, :]
        gathered = np.concatenate((q, k, v), axis=0)
        out.write(gathered.tobytes())
        return
    path = source.root / source.weight_map[ref.name]
    row_slice = getattr(ref, "row_slice", None)
    column_slice = getattr(ref, "column_slice", None)
    if row_slice is None and column_slice is None:
        # whole-tensor stream (handles f32 widening and conv squeeze)
        elements = ref.rows * ref.columns
        source_bytes = elements * BF16_BYTES
        with path.open("rb") as file:
            file.seek(offset)
            remaining = source_bytes
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                chunk = file.read(step)
                if len(chunk) != step:
                    raise PackFailure(f"short read on {ref.name}")
                remaining -= step
                if ref.weight_format == WEIGHT_BF16:
                    out.write(chunk)
                else:
                    widened = bytearray(step * 2)
                    widened[2::4] = chunk[0::2]
                    widened[3::4] = chunk[1::2]
                    out.write(widened)
        return
    full = read_source_matrix(source, ref.name)
    if ref.kind in (KIND_FINAL_NORM, KIND_MTP_HIDDEN_NORM, KIND_ATTENTION_NORM, KIND_MLP_NORM):
        # 4-stream hc norm: take the stream-0 section (receipted).
        section = full.reshape(-1)[:ref.columns]
        out.write(section.astype("<u2").tobytes())
        return
    if full.ndim == 3:
        full = full.reshape(full.shape[0], -1)
    if full.ndim == 1:
        # 1-D vectors (A_log/dt_bias) shard flat and may need f32 widening.
        if row_slice is not None:
            full = full[row_slice[0]:row_slice[0] + row_slice[1]]
        if ref.weight_format == WEIGHT_F32:
            out.write(bf16_widen(np.ascontiguousarray(full).astype("<u2").reshape(-1)).tobytes())
        else:
            out.write(np.ascontiguousarray(full).astype("<u2").tobytes())
        return
    if row_slice is not None:
        full = full[row_slice[0]:row_slice[0] + row_slice[1], :]
    if column_slice is not None:
        full = full[:, column_slice[0]:column_slice[0] + column_slice[1]]
    out.write(np.ascontiguousarray(full).astype("<u2").tobytes())


def copy_mtp_fc(source: SafetensorsSource, ref: TensorRef, out) -> None:
    import numpy as np
    embed = read_source_matrix(source, MTP_PREFIX + "fc_embedding.weight")
    hidden = read_source_matrix(source, MTP_PREFIX + "fc_hidden.weight")
    fused = np.concatenate((embed, hidden), axis=1)
    out.write(fused.astype("<u2").tobytes())


def quantize_experts(source: SafetensorsSource, ref: TensorRef, expert_format: str, out) -> None:
    """Read the fused per-layer expert tensors [E, 2I, H] / [E, H, I],
    split w1/w3 per expert, quantize per 128x128 block, and stack the
    rank's experts expert-major with the scale plane after the payload."""
    import numpy as np
    expert_start, expert_count = getattr(ref, "expert_slice", (0, EXPERT_COUNT))
    if ref.kind == KIND_MOE_DOWN:
        down = read_source_matrix(source, layer_tensor_name(ref.kind, ref.layer))
        matrix = down[expert_start:expert_start + expert_count].reshape(-1, EXPERT_INTERMEDIATE)
    else:
        gate_up = read_source_matrix(source, layer_tensor_name(KIND_MOE_W1, ref.layer))
        section = gate_up[expert_start:expert_start + expert_count]
        half = section.shape[1] // 2
        matrix = (section[:, :half, :] if ref.kind == KIND_MOE_W1
                  else section[:, half:, :]).reshape(-1, HIDDEN)
    payload, scales = quantize_fp8_blocks(bf16_to_f32_matrix(matrix), expert_format)
    out.write(payload)
    out.write(scales)


def convert(checkpoint: Path, output: Path, first_layer: int, layer_count: int,
            receipt: dict, dry_run: bool, tp_degree: int, tp_rank: int,
            expert_format: str) -> dict:
    import numpy as np  # noqa: F401  (quantization paths import lazily)
    source = SafetensorsSource(checkpoint)
    source.check_config()
    inventory = build_inventory(first_layer, layer_count)
    # Shape-validate the FULL refs against the checkpoint first and stash
    # each payload offset on the ref (shard_ref mutates in place, so the
    # attribute survives the narrowing); the TP narrowing below only
    # changes the pack-side rows/columns.
    for full_ref in inventory:
        _, _, offset = source.check_shape(full_ref)
        full_ref.source_offset = offset
    refs = [shard_ref(ref, tp_degree, tp_rank) for ref in inventory]
    # The routed experts' WIRE format (natural is F32B128; the CLI flag swaps
    # in the per-row MX plane). The plan below must price the scale bytes by
    # the WIRE format - the writer emits exactly this plane.
    expert_wire_format = WEIGHT_FP8_E8M0B128 if expert_format == "fp8-e8m0b128" else WEIGHT_FP8_F32B128
    plans = []
    cursor = 0
    for ref in refs:
        if ref.weight_format in (WEIGHT_FP8_F32B128, WEIGHT_FP8_E8M0B128):
            payload_bytes = ref.rows * ref.columns
            # F32B128: one f32 per 128x128 tile; E8M0B128: one exponent byte
            # per (row, 128-column block) - the per-row MX plane the module's
            # grouped expert kernels decode.
            scale_bytes = ((ref.rows // FP8_BLOCK) * (ref.columns // FP8_BLOCK) * F32_BYTES
                           if expert_wire_format == WEIGHT_FP8_F32B128
                           else ref.rows * (ref.columns // FP8_BLOCK))
        else:
            payload_bytes = ref.rows * ref.columns * (BF16_BYTES if ref.weight_format == WEIGHT_BF16 else F32_BYTES)
            scale_bytes = 0
        payload_offset = align_up(cursor, PAYLOAD_ALIGNMENT)
        plans.append((ref, payload_offset, payload_bytes, scale_bytes))
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
        VOCAB, MXFP4_GROUP, MTP_LAYERS,
        HEADER_BYTES, file_bytes)
    entries = b"".join(
        ENTRY_STRUCT.pack(
            ref.kind, ref.layer,
            expert_wire_format if ref.weight_format == WEIGHT_FP8_F32B128 else ref.weight_format,
            ref.rows, ref.columns,
            FP8_BLOCK if ref.weight_format in (WEIGHT_FP8_F32B128, WEIGHT_FP8_E8M0B128) else 0,
            payload_base + payload_offset, payload_bytes,
            payload_base + payload_offset + payload_bytes if scale_bytes else 0,
            scale_bytes)
        for ref, payload_offset, payload_bytes, scale_bytes in plans)
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
        print(f"qwen4_flash_stagepack slice={first_layer}+{layer_count} tp={tp_degree}/{tp_rank} "
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
        for ref, payload_offset, payload_bytes, scale_bytes in plans:
            before = temp.tell()
            if ref.kind == KIND_MTP_FC:
                copy_mtp_fc(source, ref, temp)
            elif ref.weight_format in (WEIGHT_FP8_F32B128, WEIGHT_FP8_E8M0B128):
                quantize_experts(source, ref, expert_format, temp)
            else:
                copy_sharded_bf16(source, ref, getattr(ref, "source_offset", 0), temp)
            wrote = temp.tell() - before
            if wrote != payload_bytes + scale_bytes:
                raise PackFailure(f"payload size mismatch on {ref.name}: wrote {wrote} != {payload_bytes + scale_bytes}")
            pad = align_up(temp.tell(), PAYLOAD_ALIGNMENT) - temp.tell()
            if pad:
                temp.write(b"\0" * pad)
        temp.flush()
        os.fsync(temp.fileno())
    os.replace(temp_path, output)
    receipt["output_sha256"] = sha256_file(output)
    receipt["file"] = str(output)
    print(f"qwen4_flash_stagepack slice={first_layer}+{layer_count} tp={tp_degree}/{tp_rank} "
          f"tensors={len(plans)} file_gib={file_bytes / 2**30:.2f} wrote {output}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint directory")
    parser.add_argument("--output", type=Path, help="pack output path")
    parser.add_argument("--first-layer", type=int, help="explicit slice start")
    parser.add_argument("--layer-count", type=int, help="explicit slice length")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--receipt", type=Path, help="receipt output (default: <output>.receipt.json)")
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--expert-format", choices=("fp8-f32b128", "fp8-e8m0b128"), default="fp8-f32b128")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.first_layer is None or args.layer_count is None:
        parser.error("--first-layer and --layer-count are required")
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run")

    receipt = {
        "kind": "sparkpipe.qwen4_flash.stagepack-receipt.v1",
        "tool": "tools/qwen4_flash_stagepack.py",
        "checkpoint": str(args.checkpoint),
        "contract": {"path": str(args.contract),
                     "sha256": sha256_file(args.contract) if args.contract.is_file() else None},
        "weight_formats": {"routed_experts": args.expert_format,
                           "non_expert": "bf16",
                           "gdn_a_log_dt_bias": "f32"},
        "hc_approximations": [
            "FINAL_NORM packed from hyper_connection_mixer.hc_norm stream-0 section (Flash has no plain model.norm)",
            "MTP_HIDDEN_NORM packed from mtp.pre_fc_norm_hidden stream-0 section (source is 10240-wide)",
            "MTP_FC packed as column-concat(mtp.fc_embedding, mtp.fc_hidden) [2560,5120]",
        ],
        "unmapped_checkpoint_tensors": {
            "per_layer_hyper_connection": "attn_hyper_connection/mlp_hyper_connection (4 tensors x 48 layers) + global mixer",
            "attention_indexer": "self_attn.indexer.* on 12 full-attention layers + MTP layer",
            "ple_layer_1": "layers.1.ple.* (137 tensors, ngram embedding)",
            "vision_tower": "model.visual.* (out of scope by contract)",
        },
    }
    result = convert(args.checkpoint, args.output or Path("/dev/null"),
                     args.first_layer, args.layer_count, receipt, args.dry_run,
                     args.tp_degree, args.tp_rank, args.expert_format)
    if not args.dry_run:
        receipt_path = args.receipt or Path(str(args.output) + ".receipt.json")
        write_receipt(result, receipt_path, suffix=None)
        print(f"qwen4_flash_stagepack receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen4_flash_stagepack: {error}", file=sys.stderr)
        sys.exit(1)
