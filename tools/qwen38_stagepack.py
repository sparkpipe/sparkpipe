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
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_codecs import (WEIGHT_BF16, WEIGHT_F32, WEIGHT_NVFP4_PACKED, widen_bf16_bytes)  # noqa: E402
from spark_pack_common import (GLOBAL_LAYER, HashingWriter, PackFailure,  # noqa: E402
                               SafetensorsSource as _BaseSafetensorsSource, TpPlan, align_up, packed_tp_shape,
                               pump, replaced_output, sha256_file, stream_range, tp_window, write_receipt)
from spark_pack_wire import PAYLOAD_ALIGNMENT, STAGEPACK_ENTRY, STAGEPACK_HEADER, region_offsets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "model_contracts" / "qwen38_authoritative.json"

MAGIC = 0x50533851
FORMAT_VERSION = 1
HEADER_BYTES = STAGEPACK_HEADER.size
ENTRY_BYTES = STAGEPACK_ENTRY.size
FORMAT2_VERSION = 2
HEADER2_STRUCT = struct.Struct("<28I2Q")
HEADER2_BYTES = HEADER2_STRUCT.size
MTP_LAYER = 0xFFFFFFFE
WEIGHT_FP8_F32B128 = 4
NVFP4_SEGMENT_TAIL_BYTES = 8

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
STRIP_MTP = False
EXPERT_CODEC = "fp8"
MXFP4_GROUP = 32

GDN_QK_DIM = GDN_KEY_HEADS * GDN_HEAD_KEY_DIM
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_VALUE_DIM
GDN_CONV_CHANNELS = 2 * GDN_QK_DIM + GDN_VALUE_DIM
ATTN_Q_DIM = ATTN_QUERY_HEADS * ATTN_HEAD_DIM
ATTN_KV_DIM = ATTN_KV_HEADS * ATTN_HEAD_DIM

(KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD, KIND_ATTENTION_NORM,
 KIND_MLP_NORM, KIND_MOE_GATE, KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN,
 KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_UP, KIND_MOE_SHARED_DOWN,
 KIND_MOE_SHARED_GATE_WEIGHT, KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_BETA,
 KIND_GDN_DECAY, KIND_GDN_OUTPUT, KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG,
 KIND_GDN_DT_BIAS, KIND_GDN_NORM, KIND_ATTN_QUERY, KIND_ATTN_KEY,
 KIND_ATTN_VALUE, KIND_ATTN_OUTPUT, KIND_ATTN_QUERY_NORM,
 KIND_ATTN_KEY_NORM, KIND_MTP_FC, KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM,
 KIND_MTP_FINAL_NORM) = range(32)
EXPERT_KINDS = (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN)

CHUNK_BYTES = 512 * 1024
BF16_BYTES = 2
F32_BYTES = 4

MTP_PREFIX = "mtp."


def nvfp4_experts(ref) -> bool:
    return ref.weight_format == WEIGHT_FP8_F32B128 and EXPERT_CODEC == "nvfp4"


def ref_weight_format(ref):
    if EXPERT_CODEC != "nvfp4" or ref.weight_format != WEIGHT_FP8_F32B128:
        return ref.weight_format
    if ref.kind in EXPERT_KINDS:
        return WEIGHT_BF16 if ref.layer == MTP_LAYER else WEIGHT_NVFP4_PACKED
    return ref.weight_format


def entry_group_size(ref) -> int:
    return {WEIGHT_FP8_F32B128: 128, WEIGHT_NVFP4_PACKED: 16}.get(ref_weight_format(ref), 0)


def is_gdn_layer(layer_index: int) -> bool:
    return (layer_index % ATTENTION_PERIOD) != FULL_PHASE


def kind_shape(kind: int) -> tuple[int, int, int]:
    return {
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
    }[kind]


LAYER_TENSORS = {
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
    KIND_GDN_QKV: "linear_attn.in_proj_qkv.weight",
    KIND_GDN_GATE: "linear_attn.in_proj_z.weight",
    KIND_GDN_BETA: "linear_attn.in_proj_b.weight",
    KIND_GDN_DECAY: "linear_attn.in_proj_a.weight",
    KIND_GDN_OUTPUT: "linear_attn.out_proj.weight",
    KIND_GDN_CONV_WEIGHT: "linear_attn.conv1d.weight",
    KIND_GDN_A_LOG: "linear_attn.A_log",
    KIND_GDN_DT_BIAS: "linear_attn.dt_bias",
    KIND_GDN_NORM: "linear_attn.norm.weight",
    KIND_ATTN_QUERY: "self_attn.q_proj.weight",
    KIND_ATTN_KEY: "self_attn.k_proj.weight",
    KIND_ATTN_VALUE: "self_attn.v_proj.weight",
    KIND_ATTN_OUTPUT: "self_attn.o_proj.weight",
    KIND_ATTN_QUERY_NORM: "self_attn.q_norm.weight",
    KIND_ATTN_KEY_NORM: "self_attn.k_norm.weight",
}


def layer_tensor_name(kind: int, layer: int) -> str:
    if kind not in LAYER_TENSORS:
        raise PackFailure(f"kind {kind} is not a per-layer tensor")
    return (f"model.layers.{layer}." if layer != MTP_LAYER else MTP_PREFIX + "layers.0.") + LAYER_TENSORS[kind]


GLOBAL_TENSORS = {
    KIND_EMBEDDING: "model.embed_tokens.weight",
    KIND_FINAL_NORM: "model.norm.weight",
    KIND_LM_HEAD: "lm_head.weight",
    KIND_MTP_FC: MTP_PREFIX + "fc.weight",
    KIND_MTP_EMBED_NORM: MTP_PREFIX + "pre_fc_norm_embedding.weight",
    KIND_MTP_HIDDEN_NORM: MTP_PREFIX + "pre_fc_norm_hidden.weight",
    KIND_MTP_FINAL_NORM: MTP_PREFIX + "norm.weight",
}

EVERY_LAYER_KINDS = (KIND_ATTENTION_NORM, KIND_MLP_NORM, KIND_MOE_GATE, KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN,
                     KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_UP, KIND_MOE_SHARED_DOWN, KIND_MOE_SHARED_GATE_WEIGHT)
GDN_LAYER_KINDS = (KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY, KIND_GDN_OUTPUT,
                   KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG, KIND_GDN_DT_BIAS, KIND_GDN_NORM)
ATTN_LAYER_KINDS = (KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT, KIND_ATTN_QUERY_NORM,
                    KIND_ATTN_KEY_NORM)


class TensorRef:
    def __init__(self, kind: int, layer: int, name: str):
        self.kind = kind
        self.layer = layer
        self.name = name
        self.rows, self.columns, self.weight_format = kind_shape(kind)
        self.slice_start = {KIND_MOE_W3: EXPERT_INTERMEDIATE}.get(kind, 0)


def expected_tensor_count(first_layer: int, layer_count: int) -> int:
    full = (first_layer + layer_count) // ATTENTION_PERIOD - first_layer // ATTENTION_PERIOD
    tensors = layer_count * 10 + (layer_count - full) * 9 + full * 6
    if first_layer == 0:
        tensors += 1
    if first_layer + layer_count == LAYER_COUNT:
        tensors += (2 if STRIP_MTP else 2 + 4 + 16) + (1 if first_layer != 0 else 0)
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
        tail = (KIND_FINAL_NORM, KIND_LM_HEAD) if STRIP_MTP else (KIND_FINAL_NORM, KIND_LM_HEAD, KIND_MTP_FC,
                                                                 KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM,
                                                                 KIND_MTP_FINAL_NORM)
        refs += [TensorRef(kind, GLOBAL_LAYER, GLOBAL_TENSORS[kind]) for kind in tail]
        if not STRIP_MTP:
            refs += [TensorRef(kind, MTP_LAYER, layer_tensor_name(kind, MTP_LAYER))
                     for kind in EVERY_LAYER_KINDS + ATTN_LAYER_KINDS]
    expected = expected_tensor_count(first_layer, layer_count)
    if len(refs) != expected:
        raise PackFailure(f"inventory {len(refs)} tensors, format expects {expected}")
    return refs


def fused_mtp_name(ref) -> str:
    base = ref.name.replace("{e}.", "")
    base = base[:-len(".weight")] if base.endswith(".weight") else base
    if ref.kind == KIND_MOE_DOWN:
        return base if base.endswith("down_proj") else base + ".down_proj"
    return base[:-(len("gate_proj") if ref.kind == KIND_MOE_W1 else len("up_proj"))] + "gate_up_proj"


def expert_stem(ref, expert: int) -> str:
    return ref.name.replace("{e}", str(expert))[:-len(".weight")]


class SafetensorsSource(_BaseSafetensorsSource):
    def check_config(self) -> None:
        super().check_config({
            "hidden_size": HIDDEN, "num_hidden_layers": LAYER_COUNT, "num_attention_heads": ATTN_QUERY_HEADS,
            "num_key_value_heads": ATTN_KV_HEADS, "head_dim": ATTN_HEAD_DIM, "linear_num_key_heads": GDN_KEY_HEADS,
            "linear_num_value_heads": GDN_VALUE_HEADS, "linear_key_head_dim": GDN_HEAD_KEY_DIM,
            "linear_value_head_dim": GDN_HEAD_VALUE_DIM, "linear_conv_kernel_dim": GDN_CONV_KERNEL,
            "num_experts": EXPERT_COUNT, "num_experts_per_tok": EXPERTS_PER_TOKEN,
            "moe_intermediate_size": EXPERT_INTERMEDIATE, "vocab_size": VOCAB,
            "full_attention_interval": ATTENTION_PERIOD, "attn_output_gate": True, "tie_word_embeddings": False,
        })

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        if ref.kind not in EXPERT_KINDS:
            return super().check_shape(ref.name, ref.rows, ref.columns)
        rows_per_expert = ref.rows // EXPERT_COUNT
        if EXPERT_CODEC == "nvfp4" and ref.layer == MTP_LAYER:
            fused = fused_mtp_name(ref)
            shard, meta, offset = self.resolve(fused)
            if ref.kind == KIND_MOE_DOWN:
                expect, expect3 = [ref.rows, ref.columns], [EXPERT_COUNT, rows_per_expert, ref.columns]
            else:
                expect, expect3 = [2 * ref.rows, ref.columns], [EXPERT_COUNT, 2 * EXPERT_INTERMEDIATE, ref.columns]
            if meta["dtype"] != "BF16":
                raise PackFailure(f"{fused}: dtype {meta['dtype']}, expected BF16 (fused MTP)")
            if meta["shape"] not in (expect, expect3):
                raise PackFailure(f"{fused}: checkpoint shape {meta['shape']}, pack expects {expect} "
                                  f"(fused expert-major)")
            return shard, meta, offset
        expert0 = ref.name.replace("{e}", "0")
        shard, meta, offset = self.resolve(expert0)
        if EXPERT_CODEC == "nvfp4":
            checks = ((expert0, "U8", [rows_per_expert, ref.columns // 2]),
                      (expert_stem(ref, 0) + ".weight_scale", "F8_E4M3", [rows_per_expert, ref.columns // 16]),
                      (expert_stem(ref, 0) + ".weight_scale_2", "F32", []))
        else:
            checks = ((expert0, "F8_E4M3", [rows_per_expert, ref.columns]),
                      (expert0 + "_scale_inv", "BF16", [rows_per_expert // 128, ref.columns // 128]))
        for name, dtype, shape in checks:
            got = self.resolve(name)[1]
            if got["dtype"] != dtype:
                raise PackFailure(f"{name}: dtype {got['dtype']}, expected {dtype}")
            if got["shape"] != shape:
                raise PackFailure(f"{name}: checkpoint shape {got['shape']}, pack expects {shape}")
        return shard, meta, offset


def build_tp_plan(ref, degree: int, rank: int):
    if degree <= 1 or ref.layer in (GLOBAL_LAYER, MTP_LAYER):
        return None
    if ref.kind in EXPERT_KINDS:
        offset, count = tp_window(ref.rows, degree, rank)
        return TpPlan("experts", row_off=offset, row_count=count)
    if ref.kind in (KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_UP, KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE,
                    KIND_GDN_GATE, KIND_LM_HEAD):
        offset, count = tp_window(ref.rows, degree, rank)
        return TpPlan("rows", row_off=offset, row_count=count)
    if ref.kind in (KIND_MOE_SHARED_DOWN, KIND_GDN_OUTPUT, KIND_ATTN_OUTPUT):
        offset, count = tp_window(ref.columns, degree, rank)
        return TpPlan("cols", col_off=offset, col_count=count)
    if ref.kind == KIND_GDN_QKV:
        qk = GDN_KEY_HEADS * GDN_HEAD_KEY_DIM
        value = GDN_VALUE_HEADS * GDN_HEAD_VALUE_DIM
        qk_off, qk_count = tp_window(qk, degree, rank)
        v_off, v_count = tp_window(value, degree, rank)
        return TpPlan("qkv", qk=qk, segments=((qk_off, qk_count), (qk + qk_off, qk_count),
                                              (2 * qk + v_off, v_count)))
    return None


def pumped(source: SafetensorsSource, name: str, offset: int, length: int, out) -> None:
    shard, _, base = source.resolve(name)
    with (source.root / shard).open("rb") as file:
        pump(file.fileno(), base + offset, length, out)


def streamed(source: SafetensorsSource, name: str, offset: int, length: int, out, transform=None) -> None:
    shard, _, base = source.resolve(name)
    with (source.root / shard).open("rb") as file:
        stream_range(file, base + offset, length, out, CHUNK_BYTES, f"{name}", transform)


def scale_segment(source: SafetensorsSource, ref, expert: int, out, copy) -> None:
    stem = expert_stem(ref, expert)
    copy(source, stem + ".weight_scale", 0, (ref.rows // EXPERT_COUNT) * (ref.columns // 16), out)
    for tail in (".input_scale", ".weight_scale_2"):
        copy(source, stem + tail, 0, F32_BYTES, out)


def copy_experts(source: SafetensorsSource, ref, plan, out) -> None:
    rows_per_expert = ref.rows // EXPERT_COUNT
    first, last = 0, EXPERT_COUNT
    copy = streamed
    if plan is not None:
        first = plan.row_off // rows_per_expert
        last = first + plan.row_count // rows_per_expert
        copy = pumped
    row_bytes = ref.columns // 2 if nvfp4_experts(ref) else ref.columns
    for expert in range(first, last):
        copy(source, ref.name.replace("{e}", str(expert)), 0, rows_per_expert * row_bytes, out)
    for expert in range(first, last):
        if nvfp4_experts(ref):
            scale_segment(source, ref, expert, out, copy)
        else:
            name = ref.name.replace("{e}", str(expert)) + "_scale_inv"
            streamed(source, name, 0, (rows_per_expert // 128) * (ref.columns // 128) * BF16_BYTES, out,
                     widen_bf16_bytes)


def copy_fused_mtp_experts(source: SafetensorsSource, ref, out) -> None:
    fused = fused_mtp_name(ref)
    if ref.kind == KIND_MOE_DOWN:
        streamed(source, fused, 0, ref.rows * ref.columns * BF16_BYTES, out)
        return
    row_bytes = ref.columns * BF16_BYTES
    for expert in range(EXPERT_COUNT):
        block = expert * 2 * EXPERT_INTERMEDIATE + ref.slice_start
        streamed(source, fused, block * row_bytes, EXPERT_INTERMEDIATE * row_bytes, out)


def copy_window(source: SafetensorsSource, ref, plan, out) -> None:
    row_bytes = ref.columns * BF16_BYTES
    if plan.kind == "rows":
        pumped(source, ref.name, plan.row_off * row_bytes, plan.row_count * row_bytes, out)
    elif plan.kind == "cols":
        for row in range(ref.rows):
            pumped(source, ref.name, row * row_bytes + plan.col_off * BF16_BYTES, plan.col_count * BF16_BYTES, out)
    elif plan.kind == "qkv":
        for offset, count in plan.segments:
            pumped(source, ref.name, offset * row_bytes, count * row_bytes, out)
    else:
        raise PackFailure(f"unknown tp plan {plan.kind}")


def copy_tensor(source: SafetensorsSource, ref, plan, out) -> None:
    if ref.kind in EXPERT_KINDS and (plan is not None and plan.kind == "experts" or ref.layer != MTP_LAYER
                                     or not nvfp4_experts(ref)):
        copy_experts(source, ref, plan, out)
    elif ref.kind in EXPERT_KINDS:
        copy_fused_mtp_experts(source, ref, out)
    elif plan is not None:
        copy_window(source, ref, plan, out)
    else:
        streamed(source, ref.name, 0, ref.rows * ref.columns * BF16_BYTES, out,
                 widen_bf16_bytes if ref.weight_format == WEIGHT_F32 else None)


def entry_bytes(ref, rows: int, columns: int) -> tuple[int, int]:
    if nvfp4_experts(ref) and ref.layer == MTP_LAYER:
        return rows * columns * BF16_BYTES, 0
    if nvfp4_experts(ref):
        resident = rows // (ref.rows // EXPERT_COUNT)
        return rows * (columns // 2), rows * (columns // 16) + resident * NVFP4_SEGMENT_TAIL_BYTES
    if ref.weight_format == WEIGHT_FP8_F32B128:
        return rows * columns, (rows // 128) * (columns // 128) * F32_BYTES
    return rows * columns * (BF16_BYTES if ref.weight_format == WEIGHT_BF16 else F32_BYTES), 0


def convert(checkpoint: Path, output: Path, first_layer: int, layer_count: int, receipt: dict, dry_run: bool,
            tp_degree: int = 1, tp_rank: int = 0) -> dict:
    source = SafetensorsSource(checkpoint)
    source.check_config()
    plans = []
    for ref in build_inventory(first_layer, layer_count):
        source.check_shape(ref)
        plan = build_tp_plan(ref, tp_degree, tp_rank)
        rows, columns = packed_tp_shape(ref, plan)
        plans.append((ref, plan, rows, columns) + entry_bytes(ref, rows, columns))
    offsets, end = region_offsets(((payload, scale) for *_, payload, scale in plans), 0, align_scales=False)
    mtp_layer_count = 0 if STRIP_MTP else MTP_LAYERS
    geometry = (HIDDEN, layer_count, first_layer, LAYER_COUNT, ATTENTION_PERIOD, FULL_PHASE, GDN_KEY_HEADS,
                GDN_VALUE_HEADS, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM, GDN_CONV_KERNEL, ATTN_QUERY_HEADS,
                ATTN_KV_HEADS, ATTN_HEAD_DIM, ATTN_ROPE_DIM, EXPERT_COUNT, EXPERTS_PER_TOKEN, EXPERT_INTERMEDIATE,
                VOCAB, MXFP4_GROUP, mtp_layer_count)
    if tp_degree > 1:
        payload_base = align_up(HEADER2_BYTES + len(plans) * ENTRY_BYTES, PAYLOAD_ALIGNMENT)
        header = HEADER2_STRUCT.pack(MAGIC, FORMAT2_VERSION, HEADER2_BYTES, ENTRY_BYTES, len(plans), *geometry,
                                     tp_degree, tp_rank, HEADER2_BYTES, payload_base + end)
    else:
        payload_base = align_up(HEADER_BYTES + len(plans) * ENTRY_BYTES, PAYLOAD_ALIGNMENT)
        header = STAGEPACK_HEADER.pack(MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(plans), *geometry,
                                       HEADER_BYTES, payload_base + end)
    file_bytes = payload_base + end
    entries = b"".join(
        STAGEPACK_ENTRY.pack(ref.kind, ref.layer, ref_weight_format(ref), rows, columns, entry_group_size(ref),
                             payload_base + payload_offset, payload_bytes,
                             payload_base + scale_offset if scale_bytes else 0, scale_bytes)
        for (ref, _, rows, columns, payload_bytes, scale_bytes), (payload_offset, scale_offset) in zip(plans, offsets))
    receipt.update({"first_layer_index": first_layer, "layer_count": layer_count, "tensor_count": len(plans),
                    "bytes": file_bytes, "source_index_sha256": source.index_sha256,
                    "source_config_sha256": source.config_sha256})
    if dry_run:
        print(f"qwen38_stagepack slice={first_layer}+{layer_count} tensors={len(plans)} file_bytes={file_bytes} "
              f"file_gib={file_bytes / 2**30:.2f} (dry run)")
        return receipt
    with replaced_output(output) as temp:
        hashing = HashingWriter(temp)
        hashing.write(header)
        hashing.write(entries)
        hashing.write(b"\0" * (payload_base - hashing.tell()))
        receipt.update({"tp_degree": tp_degree, "tp_rank": tp_rank})
        for ref, plan, _, _, payload_bytes, scale_bytes in plans:
            before = hashing.tell()
            copy_tensor(source, ref, plan, hashing)
            wrote = hashing.tell() - before
            if wrote != payload_bytes + scale_bytes:
                raise PackFailure(f"payload size mismatch on {ref.name}: {wrote} != {payload_bytes + scale_bytes}")
            hashing.write(b"\0" * (align_up(hashing.tell(), PAYLOAD_ALIGNMENT) - hashing.tell()))
            try:
                os.sync_file_range(temp.fileno(), before, hashing.tell() - before, 2)
                os.posix_fadvise(temp.fileno(), before, hashing.tell() - before, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
    receipt["output_sha256"] = hashing.digest.hexdigest()
    receipt["file"] = str(output)
    print(f"qwen38_stagepack slice={first_layer}+{layer_count} tensors={len(plans)} "
          f"file_gib={file_bytes / 2**30:.2f} sha256={receipt['output_sha256'][:16]}... wrote {output}")
    return receipt


def main() -> int:
    global EXPERT_CODEC, STRIP_MTP
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint directory")
    parser.add_argument("--output", type=Path, help="pack output path")
    parser.add_argument("--first-layer", type=int, help="explicit slice start")
    parser.add_argument("--layer-count", type=int, help="explicit slice length")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--receipt", type=Path, help="receipt output (default: <output>.receipt.json)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strip-mtp", action="store_true",
                        help="exclude the MTP entries (4 global kinds + the MTP-layer pseudo-layer) from the tail "
                             "stage; the pack header carries mtp_layer_count 0 and the MTP=0 module form accepts it")
    parser.add_argument("--expert-codec", choices=("fp8", "nvfp4"), default="fp8",
                        help="expert weight codec: fp8 (F8_E4M3 + BF16 scale_inv b128, the 2.3T source) or nvfp4 "
                             "(U8-packed 4-bit + F8_E4M3 g16 scales + F32 global, the radixark bf16-spine source; "
                             "payload streams rows*cols/2, scales rows*cols/16, global + input_scale in the "
                             "manifest entry)")
    parser.add_argument("--tp-degree", type=int, default=1,
                        help="tensor-parallel degree; shards the pack for --tp-rank (emits the v2 128-byte header "
                             "carrying tp_degree/tp_rank; the routed experts need the nvfp4 codec)")
    parser.add_argument("--tp-rank", type=int, default=0, help="this rank's shard (0 .. tp-degree-1)")
    args = parser.parse_args()
    EXPERT_CODEC = args.expert_codec
    STRIP_MTP = args.strip_mtp
    if args.tp_degree < 1 or args.tp_rank < 0 or args.tp_rank >= args.tp_degree:
        parser.error("--tp-rank must satisfy 0 <= tp-rank < tp-degree")
    if args.first_layer is None or args.layer_count is None:
        parser.error("--first-layer and --layer-count are required")
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run")
    for axis in (EXPERT_COUNT, GDN_VALUE_HEADS, GDN_KEY_HEADS, ATTN_QUERY_HEADS):
        if axis % args.tp_degree != 0:
            parser.error(f"tp degree {args.tp_degree} does not shard {axis} evenly")
    if ATTN_KV_HEADS % args.tp_degree != 0:
        group = ATTN_QUERY_HEADS // ATTN_KV_HEADS
        block = ATTN_QUERY_HEADS // args.tp_degree
        if args.tp_degree % ATTN_KV_HEADS != 0 or group % block != 0:
            parser.error(f"tp degree {args.tp_degree} cannot shard {ATTN_KV_HEADS} kv heads")
    if args.tp_degree > 1 and args.expert_codec == "fp8" and not args.dry_run:
        raise PackFailure("fp8 routed experts carry whole-expert BF16 scale_inv planes; a tensor-parallel pack "
                          "needs the nvfp4 codec")
    receipt = {
        "kind": "sparkpipe.qwen38.stagepack-receipt.v1",
        "tool": "tools/qwen38_stagepack.py",
        "checkpoint": str(args.checkpoint),
        "contract": {"path": str(args.contract),
                     "sha256": sha256_file(args.contract) if args.contract.is_file() else None},
        "weight_formats": {
            "routed_experts": ("nvfp4_e2m1_e4m3b16_f32tail" if args.expert_codec == "nvfp4"
                               else "fp8_e4m3_f32b128_scale_inv"),
            "non_expert": "bf16",
            "gdn_a_log_dt_bias": "f32"},
        "expert_codec": args.expert_codec,
    }
    result = convert(args.checkpoint, args.output or Path("/dev/null"), args.first_layer, args.layer_count, receipt,
                     args.dry_run, args.tp_degree, args.tp_rank)
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
