#!/usr/bin/env python3
"""Convert the Qwen/Qwen3.8-2.4T-A95B FP8 safetensors checkpoint into qwen38 stage packs.

Setup-time code, never the serving path. Mirrors the qwen38_27b packer for
the GDN/attention tensors and packs the routed-MoE inventory in the
precision ladder pinned by model_contracts/qwen38_authoritative.json:
routed experts F8_E4M3 with F32 block-128x128 scale_inv planes KEPT AS
SHIPPED, everything else BF16, A_log/dt_bias widened to F32.

Checkpoint layout pinned against the live FP8 release (per-expert tensors,
not the fused gate_up stack of a BF16 release):

  * GDN layers: input_layernorm, linear_attn.{in_proj_qkv [20480,8192] fused
    q2048|k2048|v16384, in_proj_z [16384,8192], in_proj_a/b [128,8192],
    conv1d [20480,1,4], A_log [128], dt_bias [128], norm [128],
    out_proj [8192,16384]}, post_attention_layernorm.
  * Attention layers: input_layernorm, self_attn.{q_proj [32768,8192] fused
    query|gate, k_proj/v_proj [1024,8192], o_proj, q_norm/k_norm [256]},
    post_attention_layernorm.
  * Every layer: mlp.gate.weight [512,8192], mlp.experts.{e}.gate_proj
    [2048,8192] + {e}.up_proj [2048,8192] (split here into w1/w3 stacks),
    mlp.experts.{e}.down_proj [8192,2048], each with a {e}..._scale_inv
    BF16 plane, mlp.shared_expert.{gate_proj,up_proj [2048,8192],
    down_proj [8192,2048]}, mlp.shared_expert_gate.weight [1,8192].
  * Head: lm_head [248320,8192], model.norm [8192], embed [248320,8192].
  * MTP: one decoder layer, same per-layer kinds at the MTP marker.

The vision tower is out of scope by contract.

TOPOLOGY: this packer emits WHOLE PP-STAGE slices. The qwen38_max module's
pack loader validates full-width shapes (SparkQwen38MaxModuleValidateEntry
in spark_qwen38_max_resident_decode_stage_module.c) and its TP kernels
slice heads/experts at RUN time from the resident full-width buffers
(SparkQwen38MaxLaunchGroupedExpertLinear indexes payload by tp_rank *
experts_per_rank; the attention kernels index heads/KV the same way), so
every rank of a TP group loads the SAME stage-pack file. Rank-local TP
shards of this model would fail pack_entry_invalid at load.
"""

from __future__ import annotations

import argparse
import hashlib
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
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "model_contracts" / "qwen38_authoritative.json"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

# Wire constants, mirroring spark_qwen38_stagepack_format.h.
MAGIC = 0x50533851  # 'Q8SP'
FORMAT_VERSION = 1
HEADER_BYTES = 120
ENTRY_BYTES = 56
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE
PAYLOAD_ALIGNMENT = 256

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_F32B128 = 4

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

HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES

# Tensor kinds, mirroring SparkQwen38StagePackTensorKind.
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


# (rows, columns, weight_format) per kind.
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


class TensorRef:
    def __init__(self, kind: int, layer: int, name: str):
        self.kind = kind
        self.layer = layer
        self.name = name
        self.rows, self.columns, self.weight_format = kind_shape(kind)
        # Fused gate_up_proj [E, 2*I, H] splits into W1 (rows 0:I) and
        # W3 (rows I:2I); down_proj [E, H, I] flattens with no slice.
        if kind == KIND_MOE_W1:
            self.slice_start, self.slice_rows = 0, EXPERT_INTERMEDIATE
        elif kind == KIND_MOE_W3:
            self.slice_start, self.slice_rows = EXPERT_INTERMEDIATE, EXPERT_INTERMEDIATE
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
            refs.append(TensorRef(kind, MTP_LAYER, layer_tensor_name(kind, MTP_LAYER)))
    expected = expected_tensor_count(first_layer, layer_count)
    if len(refs) != expected:
        raise PackFailure(f"inventory {len(refs)} tensors, format expects {expected}")
    return refs


class SafetensorsSource(_BaseSafetensorsSource):
    """qwen38 checkpoint reader: shared index/header/payload resolution plus
    the model's config expectations and FP8-expert shape checks."""

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
        if (ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN)
                and EXPERT_CODEC == "nvfp4" and ref.layer == MTP_LAYER):
            # radixark ships the MTP layer as FUSED BF16 spine tensors
            # (mtp.layers.0.mlp.experts.<proj>, expert-major aggregate),
            # not the per-expert nvfp4 split. Validate the fused form.
            base = ref.name.replace("{e}.", "")
            base = base[:-len(".weight")] if base.endswith(".weight") else base
            if ref.kind == KIND_MOE_DOWN:
                fused = base + ".down_proj" if not base.endswith("down_proj") else base
                shard, meta, offset = self.resolve(fused)
                expect = [ref.rows, ref.columns]
                expect3 = [EXPERT_COUNT, ref.rows // EXPERT_COUNT, ref.columns]
            else:  # W1|W3 ride the fused gate_up aggregate [2*rows, cols]
                fused = base[:-(len("gate_proj") if ref.kind == KIND_MOE_W1 else len("up_proj"))] + "gate_up_proj"
                shard, meta, offset = self.resolve(fused)
                # the fused aggregate ships either flattened [2*rows, cols]
                # or 3-D [E, 2*I, H]; both are byte-contiguous expert-major.
                expect = [2 * ref.rows, ref.columns]
                expect3 = [EXPERT_COUNT, 2 * EXPERT_INTERMEDIATE, ref.columns]
            if meta["dtype"] != "BF16":
                raise PackFailure(f"{fused}: dtype {meta['dtype']}, expected BF16 (fused MTP)")
            if meta["shape"] not in (expect, expect3):
                raise PackFailure(
                    f"{fused}: checkpoint shape {meta['shape']}, pack expects "
                    f"{expect} (fused expert-major)")
            return shard, meta, offset
        if ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN) and EXPERT_CODEC == "nvfp4":
            # radixark NVFP4: U8 payload [R, C/2] (2 values/byte), F8_E4M3
            # per-16 scales [R, C/16], F32 global + input scales (scalars).
            expert0 = ref.name.replace("{e}", "0")
            rows_per_expert = ref.rows // EXPERT_COUNT
            shard, meta, offset = self.resolve(expert0)
            if meta["dtype"] != "U8":
                raise PackFailure(f"{expert0}.weight: dtype {meta['dtype']}, expected U8 (4-bit packed)")
            if meta["shape"] != [rows_per_expert, ref.columns // 2]:
                raise PackFailure(
                    f"{expert0}.weight: checkpoint shape {meta['shape']}, pack expects "
                    f"[{rows_per_expert}, {ref.columns // 2}] (packed)")
            s_shard, s_meta, s_off = self.resolve(expert0[:-len(".weight")] + ".weight_scale")
            if s_meta["dtype"] != "F8_E4M3":
                raise PackFailure(f"{expert0}.weight_scale: dtype {s_meta['dtype']}, expected F8_E4M3")
            if s_meta["shape"] != [rows_per_expert, ref.columns // 16]:
                raise PackFailure(
                    f"{expert0}.weight_scale: checkpoint shape {s_meta['shape']}, pack expects "
                    f"[{rows_per_expert}, {ref.columns // 16}] (group 16)")
            g_shard, g_meta, g_off = self.resolve(expert0[:-len(".weight")] + ".weight_scale_2")
            if g_meta["dtype"] != "F32" or g_meta["shape"] != []:
                raise PackFailure(f"{expert0}.weight_scale_2: expected F32 scalar global scale")
            return shard, meta, offset
        if ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN):
            # per-expert FP8 tensors: validate expert 0 and the scale companion
            # (dtype AND shape; the remaining experts are resolved again per
            # expert in copy_fp8_experts, and a missing name fails there)
            expert0 = ref.name.replace("{e}", "0")
            rows_per_expert = ref.rows // EXPERT_COUNT
            shard, meta, offset = self.resolve(expert0)
            if meta["dtype"] != "F8_E4M3":
                raise PackFailure(f"{expert0}: dtype {meta['dtype']}, expected F8_E4M3")
            if meta["shape"] != [rows_per_expert, ref.columns]:
                raise PackFailure(
                    f"{expert0}: checkpoint shape {meta['shape']}, pack expects "
                    f"[{rows_per_expert}, {ref.columns}]")
            scale_name = expert0 + "_scale_inv"
            scale_shard, scale_meta, scale_offset = self.resolve(scale_name)
            if scale_meta["dtype"] != "BF16":
                raise PackFailure(f"{scale_name}: dtype {scale_meta['dtype']}, expected BF16")
            if scale_meta["shape"] != [rows_per_expert // 128, ref.columns // 128]:
                raise PackFailure(
                    f"{scale_name}: checkpoint shape {scale_meta['shape']}, pack expects "
                    f"[{rows_per_expert // 128}, {ref.columns // 128}]")
            return shard, meta, offset
        return super().check_shape(ref.name, ref.rows, ref.columns)


# -- pack writing ---------------------------------------------------------------


class _HashingWriter:
    """Write-through sha256: hashes every byte as it is written so the
    receipt's whole-file digest needs no second read pass over a finished
    multi-hundred-GiB pack (warm-storage read-back can be orders of
    magnitude slower than the write)."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data) -> int:
        written = self.stream.write(data)
        self.digest.update(data)
        return written

    def tell(self) -> int:
        return self.stream.tell()

    def flush(self) -> None:
        self.stream.flush()

    def fileno(self) -> int:
        return self.stream.fileno()


def copy_bf16_tensor(source: SafetensorsSource, ref: TensorRef, offset: int, out) -> None:
    path = source.root / source.weight_map[ref.name]
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
            else:  # f32: bf16 is the top half of f32
                widened = bytearray(step * 2)
                widened[2::4] = chunk[0::2]
                widened[3::4] = chunk[1::2]
                out.write(widened)


def copy_nvfp4_experts(source: SafetensorsSource, ref: TensorRef, out) -> None:
    """Stream per-expert NVFP4 payload [R, C/2] U8 expert-major, then the
    F8_E4M3 scale plane [E*R, C/16] byte-per-scale (the codec-6 layout;
    global + input F32 scales ride the manifest entry)."""
    experts = EXPERT_COUNT
    rows_per_expert = ref.rows // experts
    scale_cols = ref.columns // 16
    scales = bytearray()
    for e in range(experts):
        shard, meta, off = source.resolve(ref.name.replace("{e}", str(e)))
        with (source.root / shard).open("rb") as f:
            f.seek(off)
            remaining = rows_per_expert * (ref.columns // 2)
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                raw = f.read(step)
                if len(raw) != step:
                    raise PackFailure("short read on nvfp4 payload")
                remaining -= step
                out.write(raw)
        s_shard, s_meta, s_off = source.resolve(
            ref.name.replace("{e}", str(e))[:-len(".weight")] + ".weight_scale")
        with (source.root / s_shard).open("rb") as f:
            f.seek(s_off)
            sraw = f.read(rows_per_expert * scale_cols)
        if len(sraw) != rows_per_expert * scale_cols:
            raise PackFailure("short read on nvfp4 scale plane")
        scales += sraw
    out.write(bytes(scales))


def copy_fp8_experts(source: SafetensorsSource, ref: TensorRef, out) -> None:
    """Stack 512 per-expert F8_E4M3 weights and their BF16 scale_inv planes
    into the pack: payload [E*R, C] expert-major, scales [E*R/128, C/128]
    as F32 row-major (the common FP8_E4M3_F32B128 kernel layout; scale_inv
    is stored verbatim as the multiplier plane).

    Streams the payload expert by expert instead of materializing the full
    [E*R, C] tensor in memory (8 GiB per tensor on this model); the F32
    scale plane is small (rows*cols/4096*4 bytes, ~2 MiB per tensor) so it
    is buffered and appended after the payload."""
    import numpy as np
    experts = EXPERT_COUNT
    rows_per_expert = ref.rows // experts
    scale_rows = rows_per_expert // 128
    scale_cols = ref.columns // 128
    scales = bytearray()
    for e in range(experts):
        name = ref.name.replace("{e}", str(e))
        shard, meta, off = source.resolve(name)
        with (source.root / shard).open("rb") as f:
            f.seek(off)
            remaining = rows_per_expert * ref.columns
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                raw = f.read(step)
                if len(raw) != step:
                    raise PackFailure(f"short read on {name}")
                remaining -= step
                out.write(raw)
        scale_name = name + "_scale_inv"
        s_shard, s_meta, s_off = source.resolve(scale_name)
        with (source.root / s_shard).open("rb") as f:
            f.seek(s_off)
            sraw = f.read(scale_rows * scale_cols * 2)
        if len(sraw) != scale_rows * scale_cols * 2:
            raise PackFailure(f"short read on {scale_name}")
        s16 = np.frombuffer(sraw, dtype="<u2").astype(np.uint32)
        scales.extend(((s16 << 16).astype(np.uint32)).view(np.float32)
                      .astype("<f4").tobytes())
    out.write(scales)


def convert(checkpoint: Path, output: Path, first_layer: int, layer_count: int,
            receipt: dict, dry_run: bool, tp_degree: int = 1, tp_rank: int = 0) -> dict:
    source = SafetensorsSource(checkpoint)
    source.check_config()
    refs = build_inventory(first_layer, layer_count)
    plans = []
    cursor = 0
    for ref in refs:
        shard, meta, offset = source.check_shape(ref)
        plan = build_tp_plan(ref, tp_degree, tp_rank)
        pr, pc = packed_tp_shape(ref, plan)
        if (ref.weight_format == WEIGHT_FP8_F32B128 and EXPERT_CODEC == "nvfp4"
                and ref.layer == MTP_LAYER):
            payload_bytes = pr * pc * BF16_BYTES
            scale_bytes = 0
        elif ref.weight_format == WEIGHT_FP8_F32B128 and EXPERT_CODEC == "nvfp4":
            # codec-6 sizing: U8-packed payload (2 values/byte) + one F8_E4M3
            # scale byte per 16 values; F32 global/input scales ride the
            # manifest entry, not the payload stream.
            payload_bytes = pr * (pc // 2)
            scale_bytes = pr * (pc // 16)
        elif ref.weight_format == WEIGHT_FP8_F32B128:
            payload_bytes = pr * pc
            scale_bytes = (pr // 128) * (pc // 128) * F32_BYTES
        else:
            element_bytes = BF16_BYTES if ref.weight_format == WEIGHT_BF16 else F32_BYTES
            payload_bytes = pr * pc * element_bytes
            scale_bytes = 0
        payload_offset = align_up(cursor, PAYLOAD_ALIGNMENT)
        plans.append((ref, offset, payload_offset, payload_bytes, scale_bytes, plan))
        cursor = payload_offset + payload_bytes + scale_bytes
    payload_base = align_up(HEADER_BYTES + len(plans) * ENTRY_BYTES, PAYLOAD_ALIGNMENT)
    file_bytes = payload_base + cursor

    if tp_degree > 1:
        header = HEADER2_STRUCT.pack(
            MAGIC, FORMAT2_VERSION, HEADER2_BYTES, ENTRY_BYTES, len(plans),
            HIDDEN, layer_count, first_layer, LAYER_COUNT,
            ATTENTION_PERIOD, FULL_PHASE,
            GDN_KEY_HEADS, GDN_VALUE_HEADS, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM,
            GDN_CONV_KERNEL, ATTN_QUERY_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM,
            ATTN_ROPE_DIM, EXPERT_COUNT, EXPERTS_PER_TOKEN, EXPERT_INTERMEDIATE,
            VOCAB, MXFP4_GROUP, MTP_LAYERS,
            tp_degree, tp_rank, HEADER2_BYTES, file_bytes)
        payload_base = align_up(HEADER2_BYTES + len(plans) * ENTRY_BYTES, PAYLOAD_ALIGNMENT)
        file_bytes = payload_base + cursor
    else:
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
            ref.kind, ref.layer, ref.weight_format, pr, pc,
            128 if ref.weight_format == WEIGHT_FP8_F32B128 else 0,
            payload_base + payload_offset, payload_bytes,
            payload_base + payload_offset + payload_bytes if scale_bytes else 0,
            scale_bytes)
        for ref, _, payload_offset, payload_bytes, scale_bytes, plan in plans)
    receipt.update({
        "first_layer_index": first_layer,
        "layer_count": layer_count,
        "tensor_count": len(plans),
        "bytes": file_bytes,
        "source_index_sha256": source.index_sha256,
        "source_config_sha256": source.config_sha256,
    })
    if dry_run:
        print(f"qwen38_stagepack slice={first_layer}+{layer_count} "
              f"tensors={len(plans)} file_bytes={file_bytes} "
              f"file_gib={file_bytes / 2**30:.2f} (dry run)")
        return receipt

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp",
                                     dir=output.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        hashing = _HashingWriter(temp)
        hashing.write(header)
        hashing.write(entries)
        padding = payload_base - hashing.tell()
        if padding < 0:
            raise PackFailure("directory overruns the payload base")
        hashing.write(b"\0" * padding)
        receipt.update({
            "tp_degree": tp_degree,
            "tp_rank": tp_rank,
        })
        for ref, source_offset, payload_offset, payload_bytes, scale_bytes, plan in plans:
            before = hashing.tell()
            if plan is not None:
                copy_tp_plan(source, ref, plan, hashing)
            elif (ref.weight_format == WEIGHT_FP8_F32B128 and EXPERT_CODEC == "nvfp4"
                    and ref.layer == MTP_LAYER):
                base = ref.name.replace("{e}.", "")
                base = base[:-len(".weight")] if base.endswith(".weight") else base
                if ref.kind == KIND_MOE_DOWN:
                    fused = base if base.endswith("down_proj") else base + ".down_proj"
                    shard, meta, off = source.resolve(fused)
                    with (source.root / shard).open("rb") as f:
                        f.seek(off)
                        remaining = ref.rows * ref.columns * BF16_BYTES
                        while remaining > 0:
                            step = min(remaining, CHUNK_BYTES)
                            raw = f.read(step)
                            if len(raw) != step:
                                raise PackFailure(f"short read on {fused}")
                            remaining -= step
                            hashing.write(raw)
                else:
                    # fused gate_up [2*rows, cols] expert-major: each expert's
                    # block is 2*I rows; W1 takes rows 0:I, W3 rows I:2I.
                    fused = base[:-(len("gate_proj") if ref.kind == KIND_MOE_W1 else len("up_proj"))] + "gate_up_proj"
                    shard, meta, off = source.resolve(fused)
                    rows_per_expert = ref.rows // EXPERT_COUNT
                    row_bytes = ref.columns * BF16_BYTES
                    slice_rows = EXPERT_INTERMEDIATE
                    with (source.root / shard).open("rb") as f:
                        for e in range(EXPERT_COUNT):
                            block = e * 2 * EXPERT_INTERMEDIATE + ref.slice_start
                            f.seek(off + block * row_bytes)
                            remaining = slice_rows * row_bytes
                            while remaining > 0:
                                step = min(remaining, CHUNK_BYTES)
                                raw = f.read(step)
                                if len(raw) != step:
                                    raise PackFailure(f"short read on {fused} expert {e}")
                                remaining -= step
                                hashing.write(raw)
            elif ref.weight_format == WEIGHT_FP8_F32B128:
                copy_nvfp4_experts(source, ref, hashing) if EXPERT_CODEC == "nvfp4" else copy_fp8_experts(source, ref, hashing)
            else:
                copy_bf16_tensor(source, ref, source_offset, hashing)
            wrote = hashing.tell() - before
            if wrote != payload_bytes + scale_bytes:
                raise PackFailure(f"payload size mismatch on {ref.name}: {wrote} != {payload_bytes + scale_bytes}")
            pad = align_up(hashing.tell(), PAYLOAD_ALIGNMENT) - hashing.tell()
            if pad:
                hashing.write(b"\0" * pad)
        hashing.flush()
        os.fsync(temp.fileno())
    os.replace(temp_path, output)
    receipt["output_sha256"] = hashing.digest.hexdigest()
    receipt["file"] = str(output)
    print(f"qwen38_stagepack slice={first_layer}+{layer_count} tensors={len(plans)} "
          f"file_gib={file_bytes / 2**30:.2f} sha256={receipt['output_sha256'][:16]}... "
          f"wrote {output}")
    return receipt


# --- TP sharding (conforming per-rank packs; operator design law: a
# stagepack contains exactly the bytes its rank loads) -----------------

FORMAT2_VERSION = 2
HEADER2_BYTES = 128
HEADER2_STRUCT = struct.Struct("<28I2Q")
assert HEADER2_STRUCT.size == HEADER2_BYTES

def tp_window(total: int, degree: int, rank: int) -> tuple[int, int]:
    assert total % degree == 0, f"tp_window: {total} not divisible by {degree}"
    per = total // degree
    return rank * per, per

class TpPlan:
    """RowWindow / ColWindow / ExpertRange / QkvSegments / None(replicated)."""
    def __init__(self, kind: str, **kw):
        self.kind = kind
        self.__dict__.update(kw)

def build_tp_plan(ref, degree: int, rank: int):
    if degree <= 1:
        return None
    if ref.layer == GLOBAL_LAYER or ref.layer == MTP_LAYER:
        return None  # globals replicate; MTP chain rides every stage pack
    k = ref.kind
    if k in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN):
        off, count = tp_window(ref.rows, degree, rank)  # expert-major rows
        return TpPlan("experts", row_off=off, row_count=count)
    if k in (KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_UP, KIND_ATTN_QUERY,
             KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_GDN_GATE):
        off, count = tp_window(ref.rows, degree, rank)
        return TpPlan("rows", row_off=off, row_count=count)
    if k in (KIND_LM_HEAD,):
        off, count = tp_window(ref.rows, degree, rank)
        return TpPlan("rows", row_off=off, row_count=count)
    if k in (KIND_MOE_SHARED_DOWN, KIND_GDN_OUTPUT, KIND_ATTN_OUTPUT):
        off, count = tp_window(ref.columns, degree, rank)
        return TpPlan("cols", col_off=off, col_count=count)
    if k == KIND_GDN_QKV:
        qk = GDN_KEY_HEADS * GDN_HEAD_KEY_DIM
        v = GDN_VALUE_HEADS * GDN_HEAD_VALUE_DIM
        qk_off, qk_count = tp_window(qk, degree, rank)
        v_off, v_count = tp_window(v, degree, rank)
        return TpPlan("qkv", qk=qk,
                      segments=((qk_off, qk_count),
                                (qk + qk_off, qk_count),
                                (2 * qk + v_off, v_count)))
    return None  # norms, router, shared-gate-weight, conv, a_log/dt_bias, embed

def packed_tp_shape(ref, plan):
    if plan is None:
        return ref.rows, ref.columns
    if plan.kind == "rows":
        return plan.row_count, ref.columns
    if plan.kind == "cols":
        return ref.rows, plan.col_count
    if plan.kind == "experts":
        return plan.row_count, ref.columns
    if plan.kind == "qkv":
        return sum(c for _, c in plan.segments), ref.columns
    raise PackFailure(f"unknown tp plan {plan.kind}")

def copy_row_window(source: SafetensorsSource, ref, plan, out) -> None:
    """Contiguous row-axis slice of a whole-row-packed tensor."""
    shard, meta, off = source.resolve(ref.name)
    row_bytes = (ref.columns // 2) if (ref.weight_format == WEIGHT_FP8_F32B128
                                       and EXPERT_CODEC == "nvfp4") \
        else ref.columns * BF16_BYTES
    scale_row = ref.columns // 16 if ref.weight_format == WEIGHT_FP8_F32B128 else 0
    with (source.root / shard).open("rb") as f:
        f.seek(off + plan.row_off * row_bytes)
        remaining = plan.row_count * row_bytes
        while remaining > 0:
            step = min(remaining, CHUNK_BYTES)
            raw = f.read(step)
            if len(raw) != step:
                raise PackFailure(f"short read on {ref.name}")
            remaining -= step
            out.write(raw)
        if scale_row:
            s_shard, _, s_off = source.resolve(
                ref.name[:-len(".weight")] + ".weight_scale")
            f2 = (source.root / s_shard).open("rb")
            try:
                f2.seek(s_off + plan.row_off * scale_row)
                remaining = plan.row_count * scale_row
                while remaining > 0:
                    step = min(remaining, CHUNK_BYTES)
                    raw = f2.read(step)
                    if len(raw) != step:
                        raise PackFailure(f"short read on {ref.name} scales")
                    remaining -= step
                    out.write(raw)
            finally:
                f2.close()

def copy_col_window_bf16(source: SafetensorsSource, ref, plan, out) -> None:
    """Column-axis slice of a BF16 tensor (strided per row)."""
    shard, meta, off = source.resolve(ref.name)
    row_bytes = ref.columns * BF16_BYTES
    span = plan.col_count * BF16_BYTES
    with (source.root / shard).open("rb") as f:
        for r in range(ref.rows):
            f.seek(off + r * row_bytes + plan.col_off * BF16_BYTES)
            remaining = span
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                raw = f.read(step)
                if len(raw) != step:
                    raise PackFailure(f"short read on {ref.name}")
                remaining -= step
                out.write(raw)

def copy_experts_bounded(source: SafetensorsSource, ref, plan, out) -> None:
    """The per-expert nvfp4/fp8 streams, bounded to this rank's experts."""
    experts = EXPERT_COUNT
    first = plan.row_off // (ref.rows // experts)
    last = first + plan.row_count // (ref.rows // experts)
    rows_per_expert = ref.rows // experts
    scale_cols = ref.columns // 16
    scales = bytearray()
    for e in range(first, last):
        shard, meta, off = source.resolve(ref.name.replace("{e}", str(e)))
        with (source.root / shard).open("rb") as f:
            f.seek(off)
            remaining = rows_per_expert * (ref.columns // 2)
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                raw = f.read(step)
                if len(raw) != step:
                    raise PackFailure("short read on nvfp4 payload")
                remaining -= step
                out.write(raw)
        s_shard, _, s_off = source.resolve(
            ref.name.replace("{e}", str(e))[:-len(".weight")] + ".weight_scale")
        with (source.root / s_shard).open("rb") as f:
            f.seek(s_off)
            sraw = f.read(rows_per_expert * scale_cols)
        if len(sraw) != rows_per_expert * scale_cols:
            raise PackFailure("short read on nvfp4 scale plane")
        scales += sraw
    out.write(bytes(scales))

def copy_qkv_segments(source: SafetensorsSource, ref, plan, out) -> None:
    shard, meta, off = source.resolve(ref.name)
    row_bytes = ref.columns * BF16_BYTES
    with (source.root / shard).open("rb") as f:
        for seg_off, seg_count in plan.segments:
            f.seek(off + seg_off * row_bytes)
            remaining = seg_count * row_bytes
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                raw = f.read(step)
                if len(raw) != step:
                    raise PackFailure(f"short read on {ref.name}")
                remaining -= step
                out.write(raw)

def copy_tp_plan(source: SafetensorsSource, ref, plan, out) -> None:
    if plan.kind == "experts":
        copy_experts_bounded(source, ref, plan, out)
    elif plan.kind == "rows":
        copy_row_window(source, ref, plan, out)
    elif plan.kind == "cols":
        copy_col_window_bf16(source, ref, plan, out)
    elif plan.kind == "qkv":
        copy_qkv_segments(source, ref, plan, out)
    else:
        raise PackFailure(f"unknown tp plan {plan.kind}")

def main() -> int:
    global EXPERT_CODEC
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint directory")
    parser.add_argument("--output", type=Path, help="pack output path")
    parser.add_argument("--first-layer", type=int, help="explicit slice start")
    parser.add_argument("--layer-count", type=int, help="explicit slice length")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--receipt", type=Path, help="receipt output (default: <output>.receipt.json)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expert-codec", choices=("fp8", "nvfp4"), default="fp8",
        help="expert weight codec: fp8 (F8_E4M3 + BF16 scale_inv b128, the "
             "2.3T source) or nvfp4 (U8-packed 4-bit + F8_E4M3 g16 scales + "
             "F32 global, the radixark bf16-spine source; payload streams "
             "rows*cols/2, scales rows*cols/16, global + input_scale in the "
             "manifest entry)")
    parser.add_argument("--tp-degree", type=int, default=1,
                        help="tensor-parallel degree; shards the pack for --tp-rank "
                             "(emits the v2 128-byte header carrying tp_degree/tp_rank)")
    parser.add_argument("--tp-rank", type=int, default=0,
                        help="this rank's shard (0 .. tp-degree-1)")
    args = parser.parse_args()
    EXPERT_CODEC = args.expert_codec
    if args.tp_degree < 1 or args.tp_rank < 0 or args.tp_rank >= args.tp_degree:
        parser.error("--tp-rank must satisfy 0 <= tp-rank < tp-degree")

    if args.first_layer is None or args.layer_count is None:
        parser.error("--first-layer and --layer-count are required")
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run")

    receipt = {
        "kind": "sparkpipe.qwen38.stagepack-receipt.v1",
        "tool": "tools/qwen38_stagepack.py",
        "checkpoint": str(args.checkpoint),
        "contract": {"path": str(args.contract),
                     "sha256": sha256_file(args.contract) if args.contract.is_file() else None},
        "weight_formats": {"routed_experts": "fp8_e4m3_f32b128_scale_inv",
                           "non_expert": "bf16",
                           "gdn_a_log_dt_bias": "f32"},
    }
    result = convert(args.checkpoint, args.output or Path("/dev/null"),
                     args.first_layer, args.layer_count, receipt, args.dry_run,
                     args.tp_degree, args.tp_rank)
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
