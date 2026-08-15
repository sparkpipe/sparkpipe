#!/usr/bin/env python3
"""Convert the Qwen/Qwen3.8-2.4T-A95B BF16 safetensors checkpoint into qwen38 stage packs.

Setup-time code, never the serving path. Mirrors tools/qwen36_stagepack.py for
the GDN/attention tensors and adds the routed-MoE inventory with MXFP4-E2M1
expert weights (group-32 E8M0 scales), the quality-first ladder pinned by
model_contracts/qwen38_authoritative.json: routed experts MXFP4, everything
else BF16, A_log/dt_bias F32.

Checkpoint layout pinned against the HF index at revision
207bd685a7e3696cfaff12ded7c6a7ea0f88c996:

  * GDN layers: input_layernorm, linear_attn.{in_proj_qkv [20480,8192] fused
    q2048|k2048|v16384, in_proj_z [16384,8192], in_proj_a/b [128,8192],
    conv1d [20480,1,4], A_log [128], dt_bias [128], norm [128],
    out_proj [8192,16384]}, post_attention_layernorm.
  * Attention layers: input_layernorm, self_attn.{q_proj [32768,8192] fused
    query|gate, k_proj/v_proj [1024,8192], o_proj, q_norm/k_norm [256]},
    post_attention_layernorm.
  * Every layer: mlp.gate.weight [512,8192], mlp.experts.gate_up_proj
    [512,4096,8192] (fused w1|w3, split here), mlp.experts.down_proj
    [512,8192,2048], mlp.shared_expert.{gate_proj,up_proj [2048,8192],
    down_proj [8192,2048]}, mlp.shared_expert_gate.weight [1,8192].
  * Head: lm_head [248320,8192], model.norm [8192], embed [248320,8192].
  * MTP: one decoder layer, same per-layer kinds at the MTP marker.

The vision tower is out of scope by contract. TP slicing (TP4 rank-local
packs) lands in a follow-up revision once the module's shard plan is pinned;
this revision packs whole PP-stage slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

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


class PackFailure(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def align(offset: int) -> int:
    return (offset + PAYLOAD_ALIGNMENT - 1) & ~(PAYLOAD_ALIGNMENT - 1)


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
        KIND_MOE_W1: (EXPERT_COUNT * EXPERT_INTERMEDIATE, HIDDEN, WEIGHT_MXFP4_E2M1),
        KIND_MOE_W3: (EXPERT_COUNT * EXPERT_INTERMEDIATE, HIDDEN, WEIGHT_MXFP4_E2M1),
        KIND_MOE_DOWN: (EXPERT_COUNT * HIDDEN, EXPERT_INTERMEDIATE, WEIGHT_MXFP4_E2M1),
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


class SafetensorsSource:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        index_path = checkpoint / INDEX_NAME
        config_path = checkpoint / CONFIG_NAME
        if not index_path.is_file():
            raise PackFailure(f"missing {index_path}")
        if not config_path.is_file():
            raise PackFailure(f"missing {config_path}")
        self.index_sha256 = sha256_file(index_path)
        self.config_sha256 = sha256_file(config_path)
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self.config = json.loads(config_path.read_text())
        self.headers: dict[str, dict] = {}
        self.data_start: dict[str, int] = {}

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
        for key, expected in expectations.items():
            if self.config.get(key) != expected:
                raise PackFailure(
                    f"config.json {key}={self.config.get(key)!r}, expected {expected!r} "
                    "- this is not the checkpoint this packer is for")

    def shard_header(self, shard: str) -> dict:
        if shard not in self.headers:
            path = self.checkpoint / shard
            if not path.is_file():
                raise PackFailure(f"missing shard {shard}")
            with path.open("rb") as file:
                header_bytes = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(header_bytes))
            self.headers[shard] = header
            self.data_start[shard] = 8 + header_bytes
        return self.headers[shard]

    def resolve(self, name: str) -> tuple[str, dict, int]:
        if name not in self.weight_map:
            raise PackFailure(f"tensor not in checkpoint index: {name}")
        shard = self.weight_map[name]
        header = self.shard_header(shard)
        if name not in header:
            raise PackFailure(f"tensor {name} not in shard {shard}")
        meta = header[name]
        return shard, meta, self.data_start[shard] + meta["data_offsets"][0]

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        shard, meta, offset = self.resolve(ref.name)
        if meta["dtype"] != "BF16":
            raise PackFailure(f"{ref.name}: dtype {meta['dtype']}, expected BF16")
        shape = meta["shape"]
        if len(shape) == 3:
            experts, expert_rows, columns = shape
            if ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN):
                if experts != EXPERT_COUNT or columns != ref.columns:
                    raise PackFailure(
                        f"{ref.name}: 3-D shape {shape}, pack expects "
                        f"[{EXPERT_COUNT}, ?, {ref.columns}] for kind {ref.kind}")
                if ref.kind == KIND_MOE_DOWN:
                    want_rows = ref.rows // EXPERT_COUNT
                    if expert_rows != want_rows:
                        raise PackFailure(f"{ref.name}: down_proj rows {expert_rows}, expected {want_rows}")
                elif ref.slice_start + ref.slice_rows > expert_rows:
                    raise PackFailure(f"{ref.name}: slice {ref.slice_start}+{ref.slice_rows} exceeds {expert_rows}")
                return shard, meta, offset
        if len(shape) == 3 and shape[1] == 1:
            shape = [shape[0], shape[2]]
        if len(shape) == 1:
            shape = [1, shape[0]]
        if shape != [ref.rows, ref.columns]:
            raise PackFailure(
                f"{ref.name}: checkpoint shape {meta['shape']}, pack expects "
                f"[{ref.rows}, {ref.columns}] for kind {ref.kind}")
        return shard, meta, offset


# -- MXFP4-E2M1 quantization ----------------------------------------------------

# E2M1 magnitudes, OCP microscaling: exp=0 -> {0, 0.5}, exp=1 -> {1, 1.5},
# exp=2 -> {2, 3}. Scale is a power of two: value = code * 2^(e8m0 - 127).
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def quantize_mxfp4_e2m1(values: list[float]) -> tuple[bytearray, bytearray]:
    """Group-32 MXFP4-E2M1 with E8M0 scales: payload byte per element pair,
    one scale byte per 32-element group. RNE toward nearest magnitude; ties
    away from zero at the midpoint, matching cvt.rn.e2m1x2 semantics."""
    if len(values) % (2 * MXFP4_GROUP) != 0:
        raise PackFailure("mxfp4 input length must be a multiple of 64")
    payload = bytearray(len(values) // 2)
    scales = bytearray(len(values) // MXFP4_GROUP)
    for group in range(len(values) // MXFP4_GROUP):
        base = group * MXFP4_GROUP
        group_max = 0.0
        for value in values[base:base + MXFP4_GROUP]:
            magnitude = abs(value)
            if magnitude > group_max:
                group_max = magnitude
        if group_max == 0.0:
            scales[group] = 0
            continue
        # scale = 2^e with e = ceil(log2(group_max / 3)); E8M0 code = e + 127.
        exponent = math.ceil(math.log2(group_max / 3.0))
        code = max(1, min(254, exponent + 127))
        scales[group] = code
        scale = 2.0 ** (code - 127)
        for offset in range(MXFP4_GROUP):
            value = values[base + offset] / scale
            sign = 1 if value < 0.0 else 0
            magnitude = abs(value)
            best = 0
            best_distance = None
            for index, candidate in enumerate(E2M1_MAGNITUDES):
                distance = abs(magnitude - candidate)
                if best_distance is None or distance < best_distance - 1e-9:
                    best, best_distance = index, distance
                elif distance == best_distance and candidate > E2M1_MAGNITUDES[best]:
                    best = index
            # index 0 -> exp=0 m=0; 1 -> exp=0 m=1; 2 -> exp=1 m=0; 3 -> exp=1 m=1;
            # 4 -> exp=2 m=0; 5 -> exp=2 m=1.
            code_bits = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))[best]
            # 4-bit E2M1 nibble: [sign, exp1, exp0, mantissa]
            nibble = (sign << 3) | (code_bits[0] << 1) | code_bits[1]
            pair_offset = base + offset
            byte_index = pair_offset >> 1
            if pair_offset & 1:
                payload[byte_index] = (payload[byte_index] & 0x0F) | (nibble << 4)
            else:
                payload[byte_index] = (payload[byte_index] & 0xF0) | nibble
    return payload, scales


# -- pack writing ---------------------------------------------------------------


def copy_bf16_tensor(source: SafetensorsSource, ref: TensorRef, offset: int, out) -> None:
    path = source.checkpoint / source.weight_map[ref.name]
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


def copy_mxfp4_tensor(source: SafetensorsSource, ref: TensorRef, offset: int, out) -> None:
    """Read BF16 payload, quantize group-32 MXFP4-E2M1, write payload+scales.
    The routed expert tensors are 3-D [E, R, C] in the checkpoint; the pack
    flattens them to [E*slice_rows, C] (W1/W3 take their half of the fused
    gate_up rows per expert)."""
    import numpy as np
    path = source.checkpoint / source.weight_map[ref.name]
    row_bytes = ref.columns * BF16_BYTES
    experts = ref.rows // (ref.slice_rows or ref.columns or 1)
    if ref.slice_rows == 0:
        experts = ref.rows // ref.columns
    with path.open("rb") as file:
        for expert in range(experts):
            if ref.slice_rows:
                rows = ref.slice_rows
                row_start = ref.slice_start
                total_rows = (ref.rows // experts)
            else:
                rows = ref.rows // experts
                row_start = 0
                total_rows = rows
            for row in range(rows):
                file.seek(offset + ((expert * total_rows + row_start + row) * row_bytes))
                raw = file.read(row_bytes)
                if len(raw) != row_bytes:
                    raise PackFailure(f"short read on {ref.name} expert {expert} row {row}")
                values = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
                f32 = ((values << 16).astype(np.uint32)).view(np.float32).astype(np.float64).tolist()
                payload, scales = quantize_mxfp4_e2m1(f32)
                out.write(payload)
                out.write(scales)


def convert(checkpoint: Path, output: Path, first_layer: int, layer_count: int,
            receipt: dict, dry_run: bool) -> dict:
    source = SafetensorsSource(checkpoint)
    source.check_config()
    refs = build_inventory(first_layer, layer_count)
    plans = []
    cursor = 0
    for ref in refs:
        shard, meta, offset = source.check_shape(ref)
        if ref.weight_format == WEIGHT_MXFP4_E2M1:
            payload_bytes = ref.rows * ref.columns // 2
            scale_bytes = ref.rows * ref.columns // MXFP4_GROUP
        else:
            payload_bytes = ref.rows * ref.columns * (BF16_BYTES if ref.weight_format == WEIGHT_BF16 else F32_BYTES)
            scale_bytes = 0
        payload_offset = align(cursor)
        plans.append((ref, offset, payload_offset, payload_bytes, scale_bytes))
        cursor = payload_offset + payload_bytes + scale_bytes
    payload_base = align(HEADER_BYTES + len(plans) * ENTRY_BYTES)
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
            ref.kind, ref.layer, ref.weight_format, ref.rows, ref.columns,
            MXFP4_GROUP if ref.weight_format == WEIGHT_MXFP4_E2M1 else 0,
            payload_base + payload_offset, payload_bytes,
            payload_base + payload_offset + payload_bytes if scale_bytes else 0,
            scale_bytes)
        for ref, _, payload_offset, payload_bytes, scale_bytes in plans)
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
        temp.write(header)
        temp.write(entries)
        padding = payload_base - temp.tell()
        if padding < 0:
            raise PackFailure("directory overruns the payload base")
        temp.write(b"\0" * padding)
        for ref, source_offset, payload_offset, payload_bytes, scale_bytes in plans:
            before = temp.tell()
            if ref.weight_format == WEIGHT_MXFP4_E2M1:
                copy_mxfp4_tensor(source, ref, source_offset, temp)
            else:
                copy_bf16_tensor(source, ref, source_offset, temp)
            wrote = temp.tell() - before
            if wrote != payload_bytes + scale_bytes:
                raise PackFailure(f"payload size mismatch on {ref.name}: {wrote} != {payload_bytes + scale_bytes}")
            pad = align(temp.tell()) - temp.tell()
            if pad:
                temp.write(b"\0" * pad)
        temp.flush()
        os.fsync(temp.fileno())
    os.replace(temp_path, output)
    receipt["output_sha256"] = sha256_file(output)
    receipt["file"] = str(output)
    print(f"qwen38_stagepack slice={first_layer}+{layer_count} tensors={len(plans)} "
          f"file_gib={file_bytes / 2**30:.2f} wrote {output}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint directory")
    parser.add_argument("--output", type=Path, help="pack output path")
    parser.add_argument("--first-layer", type=int, help="explicit slice start")
    parser.add_argument("--layer-count", type=int, help="explicit slice length")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--receipt", type=Path, help="receipt output (default: <output>.receipt.json)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
        "weight_formats": {"routed_experts": "mxfp4_e2m1",
                           "non_expert": "bf16",
                           "gdn_a_log_dt_bias": "f32"},
    }
    result = convert(args.checkpoint, args.output or Path("/dev/null"),
                     args.first_layer, args.layer_count, receipt, args.dry_run)
    if not args.dry_run:
        receipt_path = args.receipt or Path(str(args.output) + ".receipt.json")
        receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"qwen38_stagepack receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen38_stagepack: {error}", file=sys.stderr)
        sys.exit(1)
