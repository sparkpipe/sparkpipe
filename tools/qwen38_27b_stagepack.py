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
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_codecs import (WEIGHT_BF16, WEIGHT_F32, WEIGHT_NVFP4_PACKED, bf16_bits_to_float,  # noqa: E402
                               float_to_bf16_bits, widen_bf16_bytes)
from spark_pack_common import (GLOBAL_LAYER, PackFailure, SafetensorsSource as _BaseSafetensorsSource,  # noqa: E402
                               TpPlan, align_up, packed_tp_shape, replaced_output, sha256_file,
                               spark_pack_replicated_draft_rows, tp_window, window_chunks, write_receipt)
from spark_pack_wire import PAYLOAD_ALIGNMENT, STAGEPACK_ENTRY, STAGEPACK_HEADER, region_offsets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "model_contracts" / "qwen38_27b_authoritative.json"

MAGIC = 0x50533651
FORMAT_VERSION = 3
HEADER_BYTES = STAGEPACK_HEADER.size
ENTRY_BYTES = STAGEPACK_ENTRY.size
MTP_LAYER = 0xFFFFFFFE
WEIGHT_FP8_E4M3_F32B128 = 5
NVFP4_GROUP = 16
FP8_SCALE_GROUP = 128

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

GDN_QK_DIM = GDN_KEY_HEADS * GDN_HEAD_KEY_DIM
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_VALUE_DIM
GDN_CONV_CHANNELS = 2 * GDN_QK_DIM + GDN_VALUE_DIM
ATTN_Q_DIM = ATTN_QUERY_HEADS * ATTN_HEAD_DIM
ATTN_KV_DIM = ATTN_KV_HEADS * ATTN_HEAD_DIM

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


def is_gdn_layer(layer_index: int) -> bool:
    return (layer_index % ATTENTION_PERIOD) != FULL_PHASE


def kind_shape(kind: int) -> tuple[int, int, int]:
    return {
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
    }[kind]


GDN_TENSORS = {
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
ATTN_TENSORS = {
    KIND_ATTN_QUERY: "self_attn.q_proj.weight",
    KIND_ATTN_KEY: "self_attn.k_proj.weight",
    KIND_ATTN_VALUE: "self_attn.v_proj.weight",
    KIND_ATTN_OUTPUT: "self_attn.o_proj.weight",
    KIND_ATTN_QUERY_NORM: "self_attn.q_norm.weight",
    KIND_ATTN_KEY_NORM: "self_attn.k_norm.weight",
}
EVERY_TENSORS = {
    KIND_ATTENTION_NORM: "input_layernorm.weight",
    KIND_MLP_NORM: "post_attention_layernorm.weight",
    KIND_FFN_GATE: "mlp.gate_proj.weight",
    KIND_FFN_UP: "mlp.up_proj.weight",
    KIND_FFN_DOWN: "mlp.down_proj.weight",
}


def layer_tensor_name(kind: int, layer: int) -> str:
    prefix = (LANGUAGE_PREFIX + f"layers.{layer}.") if layer != MTP_LAYER else (MTP_PREFIX + "layers.0.")
    for mapping in (EVERY_TENSORS, GDN_TENSORS, ATTN_TENSORS):
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

EVERY_LAYER_KINDS = (KIND_ATTENTION_NORM, KIND_MLP_NORM, KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN)
GDN_LAYER_KINDS = (KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY, KIND_GDN_OUTPUT,
                   KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG, KIND_GDN_DT_BIAS, KIND_GDN_NORM)
ATTN_LAYER_KINDS = (KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT, KIND_ATTN_QUERY_NORM,
                    KIND_ATTN_KEY_NORM)
FP8_KINDS = frozenset([KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN, KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_OUTPUT,
                       KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT])
NORM_PLUS_ONE_KINDS = frozenset([KIND_ATTENTION_NORM, KIND_MLP_NORM, KIND_FINAL_NORM, KIND_ATTN_QUERY_NORM,
                                 KIND_ATTN_KEY_NORM, KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM])
QWEN_MTP_KINDS = frozenset((KIND_MTP_FC, KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM))


class TensorRef:
    def __init__(self, kind: int, layer: int, name: str):
        self.kind = kind
        self.layer = layer
        self.name = name
        self.rows, self.columns, self.weight_format = kind_shape(kind)
        self.scale_name = None
        self.fused_up_offset = 0


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
        for kind in (KIND_FINAL_NORM, KIND_LM_HEAD, KIND_MTP_FC, KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM,
                     KIND_MTP_FINAL_NORM):
            refs.append(TensorRef(kind, GLOBAL_LAYER, GLOBAL_TENSORS[kind]))
        for kind in EVERY_LAYER_KINDS + ATTN_LAYER_KINDS:
            refs.append(TensorRef(kind, MTP_LAYER, layer_tensor_name(kind, MTP_LAYER)))
    expected = expected_tensor_count(first_layer, layer_count)
    if len(refs) != expected:
        raise PackFailure(f"inventory {len(refs)} tensors, format expects {expected}")
    return refs


def expected_tensor_count(first_layer: int, layer_count: int) -> int:
    full = (first_layer + layer_count) // ATTENTION_PERIOD - first_layer // ATTENTION_PERIOD
    tensors = layer_count * 5 + (layer_count - full) * 9 + full * 6
    if first_layer == 0:
        tensors += 1
    if first_layer + layer_count == LAYER_COUNT:
        tensors += 2 + 4 + 11 + (1 if first_layer != 0 else 0)
    return tensors


class SafetensorsSource(_BaseSafetensorsSource):
    def check_config(self) -> None:
        super().check_config({
            "hidden_size": HIDDEN, "num_hidden_layers": LAYER_COUNT, "num_attention_heads": ATTN_QUERY_HEADS,
            "num_key_value_heads": ATTN_KV_HEADS, "head_dim": ATTN_HEAD_DIM, "linear_num_key_heads": GDN_KEY_HEADS,
            "linear_num_value_heads": GDN_VALUE_HEADS, "linear_key_head_dim": GDN_HEAD_KEY_DIM,
            "linear_value_head_dim": GDN_HEAD_VALUE_DIM, "linear_conv_kernel_dim": GDN_CONV_KERNEL,
            "intermediate_size": FFN_INTERMEDIATE, "vocab_size": VOCAB, "full_attention_interval": ATTENTION_PERIOD,
            "attn_output_gate": True, "tie_word_embeddings": False,
        }, section="text_config")
        layer_types = self.config.get("text_config", {}).get("layer_types", [])
        if len(layer_types) == LAYER_COUNT:
            for layer, layer_type in enumerate(layer_types):
                want = "linear_attention" if is_gdn_layer(layer) else "full_attention"
                if layer_type != want:
                    raise PackFailure(f"config layer_types[{layer}]={layer_type!r}, expected {want!r}")

    def check_shape_nvfp4a16(self, ref: TensorRef):
        stem = ref.name[:-len(".weight")]
        shard, meta, offset = self.resolve(stem + ".weight_packed")
        if meta["dtype"] != "U8" or meta["shape"] != [ref.rows, ref.columns // 2]:
            raise PackFailure(f"{stem}.weight_packed: {meta['dtype']} {meta['shape']}, expected U8 "
                              f"[{ref.rows},{ref.columns // 2}]")
        scale_meta = self.resolve(stem + ".weight_scale")[1]
        if scale_meta["dtype"] != "F8_E4M3" or scale_meta["shape"] != [ref.rows, ref.columns // 16]:
            raise PackFailure(f"{stem}.weight_scale: {scale_meta['dtype']} {scale_meta['shape']}, expected F8_E4M3 "
                              f"[{ref.rows},{ref.columns // 16}]")
        global_meta = self.resolve(stem + ".weight_global_scale")[1]
        if global_meta["dtype"] != "F32" or global_meta["shape"] not in ([], [1]):
            raise PackFailure(f"{stem}.weight_global_scale: {global_meta['dtype']} {global_meta['shape']}, "
                              f"expected F32 scalar")
        ref.weight_format = WEIGHT_NVFP4_PACKED
        ref.name = stem + ".weight_packed"
        ref.scale_name = stem + ".weight_scale"
        ref.nvfp4_global_name = stem + ".weight_global_scale"
        return shard, meta, offset

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        actual_dtype = self.resolve(ref.name)[1]["dtype"]
        if ref.kind in (KIND_FFN_GATE, KIND_FFN_UP):
            gate_name = ref.name.rsplit(".up_proj.", 1)[0] + ".gate_proj.weight" if ref.kind == KIND_FFN_UP else ref.name
            if self.resolve(gate_name)[1]["shape"][0] == 2 * FFN_INTERMEDIATE:
                ref.fused_up_offset = FFN_INTERMEDIATE if ref.kind == KIND_FFN_UP else 0
                ref.name = gate_name
        if ref.kind in FP8_KINDS and actual_dtype == "F8_E4M3":
            ref.weight_format = WEIGHT_FP8_E4M3_F32B128
            ref.scale_name = ref.name + "_scale_inv"
        else:
            ref.scale_name = None
        return super().check_shape(ref.name, ref.rows, ref.columns, dtype=actual_dtype)


class Nvfp4A16Source(SafetensorsSource):
    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        if ref.kind in (KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN) and ref.layer != MTP_LAYER:
            return self.check_shape_nvfp4a16(ref)
        return super().check_shape(ref)


def build_tp_plan(ref, degree, rank):
    if degree <= 1:
        return None
    if spark_pack_replicated_draft_rows(ref.kind, ref.layer, draft_layer_first=MTP_LAYER,
                                        draft_layer_count=MTP_LAYERS, global_kinds=QWEN_MTP_KINDS,
                                        draft_layer_kinds_slice=True):
        return None
    rows = {KIND_LM_HEAD: VOCAB, KIND_FFN_GATE: FFN_INTERMEDIATE, KIND_FFN_UP: FFN_INTERMEDIATE,
            KIND_ATTN_QUERY: 2 * ATTN_Q_DIM, KIND_ATTN_KEY: ATTN_KV_DIM, KIND_ATTN_VALUE: ATTN_KV_DIM}
    columns = {KIND_FFN_DOWN: FFN_INTERMEDIATE, KIND_GDN_OUTPUT: GDN_VALUE_DIM, KIND_ATTN_OUTPUT: ATTN_Q_DIM}
    if ref.layer == GLOBAL_LAYER and ref.kind != KIND_LM_HEAD:
        return None
    if ref.kind in rows:
        offset, count = tp_window(rows[ref.kind], degree, rank)
        return TpPlan("rows", row_off=offset, row_count=count)
    if ref.kind in columns:
        offset, count = tp_window(columns[ref.kind], degree, rank)
        return TpPlan("cols", col_off=offset, col_count=count)
    if ref.kind == KIND_GDN_QKV:
        qk_off, qk_count = tp_window(GDN_QK_DIM, degree, rank)
        v_off, v_count = tp_window(GDN_VALUE_DIM, degree, rank)
        return TpPlan("qkv", segments=[(qk_off, qk_count), (GDN_QK_DIM + qk_off, qk_count),
                                       (2 * GDN_QK_DIM + v_off, v_count)])
    return None


def fold_plus_one(chunk: bytes) -> bytes:
    count = len(chunk) // BF16_BYTES
    values = struct.unpack("<" + "H" * count, chunk)
    return struct.pack("<" + "H" * count, *(float_to_bf16_bits(bf16_bits_to_float(value) + 1.0) for value in values))


def copy_window(path: Path, base: int, row_bytes: int, rows, column_offset: int, column_bytes: int, out,
                transform=None) -> None:
    with open(path, "rb") as file:
        read_at = lambda offset, length: os.pread(file.fileno(), length, offset)
        for row_start, row_count in rows:
            for chunk in window_chunks(read_at, base, row_bytes, row_start, row_count, column_offset, column_bytes,
                                       CHUNK_BYTES, str(path)):
                out.write(chunk if transform is None else transform(bytes(chunk)))


def row_spans(ref, plan, rows: int):
    if plan is None or plan.kind == "cols":
        return [(0, rows)]
    if plan.kind == "qkv":
        return plan.segments
    return [(plan.row_off, plan.row_count)]


def copy_tensor(source: SafetensorsSource, ref: TensorRef, offset: int, plan, out) -> None:
    path = source.root / source.weight_map[ref.name]
    if ref.weight_format == WEIGHT_NVFP4_PACKED:
        row_bytes = ref.columns // 2
        column = (plan.col_off // 2, plan.col_count // 2) if plan is not None and plan.kind == "cols" else (0, row_bytes)
        copy_window(path, offset, row_bytes, row_spans(ref, plan, ref.rows), *column, out)
        return
    element = 1 if ref.weight_format == WEIGHT_FP8_E4M3_F32B128 else BF16_BYTES
    transform = None
    if ref.weight_format == WEIGHT_F32:
        transform = widen_bf16_bytes
    elif ref.weight_format != WEIGHT_FP8_E4M3_F32B128 and ref.kind in NORM_PLUS_ONE_KINDS:
        transform = fold_plus_one
    row_bytes = ref.columns * element
    column = (plan.col_off * element, plan.col_count * element) if plan is not None and plan.kind == "cols" else (0, row_bytes)
    base = offset + (ref.fused_up_offset * row_bytes if plan is not None and plan.kind != "qkv" else 0)
    copy_window(path, base, row_bytes, row_spans(ref, plan, ref.rows), *column, out, transform)


def copy_scale(source: SafetensorsSource, ref: TensorRef, plan, out) -> None:
    path = source.root / source.weight_map[ref.scale_name]
    _, _, scale_base = source.resolve(ref.scale_name)
    if ref.weight_format == WEIGHT_NVFP4_PACKED:
        plane_columns = ref.columns // 16
        column = (plan.col_off // 16, plan.col_count // 16) if plan is not None and plan.kind == "cols" else (0, plane_columns)
        copy_window(path, scale_base, plane_columns, row_spans(ref, plan, ref.rows), *column, out)
        global_shard, _, global_offset = source.resolve(ref.nvfp4_global_name)
        with open(source.root / global_shard, "rb") as file:
            file.seek(global_offset)
            raw_global = struct.unpack("<f", file.read(4))[0]
        out.write(struct.pack("<f", 1.0 / raw_global))
        return
    scale_columns = ref.columns // FP8_SCALE_GROUP
    row_bytes = scale_columns * BF16_BYTES
    if plan is not None and plan.kind == "qkv":
        spans = [(row_off // FP8_SCALE_GROUP, (row_off + count) // FP8_SCALE_GROUP - row_off // FP8_SCALE_GROUP)
                 for row_off, count in plan.segments]
        copy_window(path, scale_base, row_bytes, spans, 0, row_bytes, out, widen_bf16_bytes)
        return
    row_off, row_count, col_off, col_count = 0, ref.rows, 0, ref.columns
    if plan is not None and plan.kind == "rows":
        row_off, row_count = plan.row_off, plan.row_count
    elif plan is not None and plan.kind == "cols":
        col_off, col_count = plan.col_off, plan.col_count
    fused = ref.fused_up_offset if plan is not None else 0
    first = (row_off + fused) // FP8_SCALE_GROUP
    last = (row_off + row_count + fused) // FP8_SCALE_GROUP
    c0, c1 = col_off // FP8_SCALE_GROUP, (col_off + col_count) // FP8_SCALE_GROUP
    copy_window(path, scale_base, row_bytes, [(first, last - first)], c0 * BF16_BYTES, (c1 - c0) * BF16_BYTES, out,
                widen_bf16_bytes)


def entry_bytes(ref: TensorRef, rows: int, columns: int) -> tuple[int, int]:
    if ref.weight_format == WEIGHT_FP8_E4M3_F32B128:
        return rows * columns, (rows // FP8_SCALE_GROUP) * (columns // FP8_SCALE_GROUP) * F32_BYTES
    if ref.weight_format == WEIGHT_NVFP4_PACKED:
        return rows * (columns // 2), rows * (columns // 16) + 4
    return rows * columns * (BF16_BYTES if ref.weight_format == WEIGHT_BF16 else F32_BYTES), 0


def scale_group(weight_format: int) -> int:
    return {WEIGHT_NVFP4_PACKED: NVFP4_GROUP, WEIGHT_FP8_E4M3_F32B128: FP8_SCALE_GROUP}.get(weight_format, 0)


def convert(checkpoint: Path, output: Path, first_layer: int, layer_count: int, receipt: dict, dry_run: bool,
            tp_degree: int = 1, tp_rank: int = 0, ffn_format: str = "bf16") -> dict:
    source = Nvfp4A16Source(checkpoint) if ffn_format == "nvfp4a16" else SafetensorsSource(checkpoint)
    source.check_config()
    plans = []
    for ref in build_inventory(first_layer, layer_count):
        _, _, offset = source.check_shape(ref)
        plan = build_tp_plan(ref, tp_degree, tp_rank)
        rows, columns = packed_tp_shape(ref, plan)
        plans.append((ref, offset, plan, rows, columns) + entry_bytes(ref, rows, columns))
    offsets, end = region_offsets(((payload, scale) for *_, payload, scale in plans), 0)
    payload_base = align_up(HEADER_BYTES + len(plans) * ENTRY_BYTES, PAYLOAD_ALIGNMENT)
    file_bytes = payload_base + end
    header = STAGEPACK_HEADER.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(plans), HIDDEN, layer_count, first_layer, LAYER_COUNT,
        ATTENTION_PERIOD, FULL_PHASE, GDN_KEY_HEADS, GDN_VALUE_HEADS, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM,
        GDN_CONV_KERNEL, ATTN_QUERY_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM, ATTN_ROPE_DIM, FFN_INTERMEDIATE, VOCAB,
        MXFP4_GROUP, MTP_LAYERS, tp_degree, tp_rank, HEADER_BYTES, file_bytes)
    entries = b"".join(
        STAGEPACK_ENTRY.pack(ref.kind, ref.layer, ref.weight_format, rows, columns, scale_group(ref.weight_format),
                             payload_base + payload_offset, payload_bytes,
                             payload_base + scale_offset if scale_bytes else 0, scale_bytes)
        for (ref, _, _, rows, columns, payload_bytes, scale_bytes), (payload_offset, scale_offset)
        in zip(plans, offsets))
    receipt.update({"first_layer_index": first_layer, "layer_count": layer_count, "tensor_count": len(plans),
                    "bytes": file_bytes, "tp_degree": tp_degree, "tp_rank": tp_rank,
                    "source_index_sha256": source.index_sha256, "source_config_sha256": source.config_sha256})
    if dry_run:
        print(f"qwen38_27b_stagepack slice={first_layer}+{layer_count} tensors={len(plans)} file_bytes={file_bytes} "
              f"file_gib={file_bytes / 2**30:.2f} (dry run)")
        return receipt
    with replaced_output(output) as out:
        out.write(header)
        out.write(entries)
        out.write(b"\0" * (payload_base - out.tell()))
        for (ref, offset, plan, _, _, payload_bytes, scale_bytes), (payload_offset, scale_offset) in zip(plans, offsets):
            before = out.tell()
            copy_tensor(source, ref, offset, plan, out)
            if out.tell() - before != payload_bytes:
                raise PackFailure(f"payload size mismatch on {ref.name}")
            if scale_bytes:
                if out.tell() != payload_base + scale_offset:
                    raise PackFailure(f"scale offset drift on {ref.name}")
                copy_scale(source, ref, plan, out)
                if out.tell() - (payload_base + scale_offset) != scale_bytes:
                    raise PackFailure(f"scale size mismatch on {ref.name}")
            out.write(b"\0" * (align_up(out.tell(), PAYLOAD_ALIGNMENT) - out.tell()))
    receipt["output_sha256"] = sha256_file(output)
    receipt["file"] = str(output)
    print(f"qwen38_27b_stagepack slice={first_layer}+{layer_count} tensors={len(plans)} "
          f"file_gib={file_bytes / 2**30:.2f} wrote {output}")
    return receipt


class PlanShape:
    def __init__(self, kind: int, layer: int, rows: int, columns: int):
        self.kind, self.layer, self.rows, self.columns = kind, layer, rows, columns


def verify(pack_path: Path) -> dict:
    file_bytes = pack_path.stat().st_size
    with pack_path.open("rb") as file:
        raw_header = file.read(HEADER_BYTES)
        if len(raw_header) != HEADER_BYTES:
            raise PackFailure("short header")
        (magic, version, header_bytes, entry_bytes_, tensor_count, hidden, layer_count, first_layer, total_layers,
         period, phase, gdn_kh, gdn_vh, gdn_kd, gdn_vd, conv_kernel, attn_qh, attn_kvh, attn_hd, rope_dim, ffn, vocab,
         mxfp4, mtp, tp_degree, tp_rank, directory_offset, declared_bytes) = STAGEPACK_HEADER.unpack(raw_header)
        geometry = {
            "magic": (magic, MAGIC), "format_version": (version, FORMAT_VERSION),
            "header_bytes": (header_bytes, HEADER_BYTES), "directory_entry_bytes": (entry_bytes_, ENTRY_BYTES),
            "tensor_count": (tensor_count, expected_tensor_count(first_layer, layer_count)),
            "hidden_dimension": (hidden, HIDDEN), "layer_count": (layer_count, layer_count),
            "total_layer_count": (total_layers, LAYER_COUNT), "attention_period": (period, ATTENTION_PERIOD),
            "full_attention_phase": (phase, FULL_PHASE), "gdn_key_head_count": (gdn_kh, GDN_KEY_HEADS),
            "gdn_value_head_count": (gdn_vh, GDN_VALUE_HEADS), "gdn_head_key_dimension": (gdn_kd, GDN_HEAD_KEY_DIM),
            "gdn_head_value_dimension": (gdn_vd, GDN_HEAD_VALUE_DIM), "gdn_conv_kernel": (conv_kernel, GDN_CONV_KERNEL),
            "attn_query_head_count": (attn_qh, ATTN_QUERY_HEADS), "attn_kv_head_count": (attn_kvh, ATTN_KV_HEADS),
            "attn_head_dimension": (attn_hd, ATTN_HEAD_DIM), "attn_rope_dimension": (rope_dim, ATTN_ROPE_DIM),
            "ffn_intermediate_dimension": (ffn, FFN_INTERMEDIATE), "output_vocab_count": (vocab, VOCAB),
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
        cursor = align_up(HEADER_BYTES + tensor_count * ENTRY_BYTES, PAYLOAD_ALIGNMENT)
        seen = set()
        for index in range(tensor_count):
            (kind, layer, weight_format, rows, columns, group, payload_offset, payload_bytes, scale_offset,
             scale_bytes) = STAGEPACK_ENTRY.unpack_from(raw_directory, index * ENTRY_BYTES)
            expected_rows, expected_columns, expected_format = kind_shape(kind)
            is_fp8 = weight_format == WEIGHT_FP8_E4M3_F32B128
            is_nvfp4 = weight_format == WEIGHT_NVFP4_PACKED
            if tp_degree > 1:
                shape = PlanShape(kind, layer, expected_rows, expected_columns)
                expected_rows, expected_columns = packed_tp_shape(shape, build_tp_plan(shape, tp_degree, tp_rank))
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
            if payload_offset != align_up(cursor, PAYLOAD_ALIGNMENT) or payload_offset % PAYLOAD_ALIGNMENT != 0:
                raise PackFailure(f"entry {index}: payload offset {payload_offset}, expected "
                                  f"{align_up(cursor, PAYLOAD_ALIGNMENT)}")
            if is_fp8:
                if (group != FP8_SCALE_GROUP or scale_offset == 0
                        or scale_bytes != (rows // FP8_SCALE_GROUP) * (columns // FP8_SCALE_GROUP) * F32_BYTES):
                    raise PackFailure(f"entry {index}: FP8 scale metadata mismatch")
            elif is_nvfp4:
                if group != NVFP4_GROUP or scale_offset == 0:
                    raise PackFailure(f"entry {index}: nvfp4 scale metadata mismatch")
                if scale_bytes != rows * (columns // 16) + 4:
                    raise PackFailure(f"entry {index}: nvfp4 scale byte count mismatch")
            elif group != 0 or scale_offset != 0 or scale_bytes != 0:
                raise PackFailure(f"entry {index}: BF16/F32 tensors carry no scales")
            if is_nvfp4:
                expected_payload = rows * (columns // 2)
            else:
                expected_payload = rows * columns * (1 if is_fp8 else BF16_BYTES if weight_format == WEIGHT_BF16
                                                     else F32_BYTES)
            if payload_bytes != expected_payload:
                raise PackFailure(f"entry {index}: payload byte count mismatch")
            cursor = (scale_offset + scale_bytes) if (is_fp8 or is_nvfp4) else (payload_offset + payload_bytes)
        if align_up(cursor, PAYLOAD_ALIGNMENT) != align_up(file_bytes, PAYLOAD_ALIGNMENT):
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
    parser.add_argument("--tp-degree", type=int, default=1, help="tensor-parallel degree; shards the pack for --tp-rank")
    parser.add_argument("--tp-rank", type=int, default=0, help="this rank's shard (0 .. tp-degree-1)")
    parser.add_argument("--ffn-format", choices=("bf16", "nvfp4a16"), default="bf16",
                        help="nvfp4a16 = the main-layer FFN projections from the nvfp4a16-bf16-spine release on the "
                             "NVFP4 wire (weight-only, BF16 activations); default bf16")
    args = parser.parse_args()
    if args.tp_degree < 1 or args.tp_rank < 0 or args.tp_rank >= args.tp_degree:
        parser.error("--tp-rank must satisfy 0 <= tp-rank < tp-degree")
    if args.tp_degree > 1 and (args.first_layer != 0 or args.layer_count != LAYER_COUNT):
        print(f"qwen38_27b_stagepack: combined TP{args.tp_degree}+PP slice {args.first_layer}+{args.layer_count} "
              f"(module needs the TP4PP4 rank map; single-pack full-stack loaders will reject it)")
    if args.verify is not None:
        result = verify(args.verify)
        print(f"qwen38_27b_stagepack verify ok: {result['file']} slice={result['first_layer_index']}+"
              f"{result['layer_count']} tensors={result['tensor_count']} bytes={result['bytes']}")
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
        receipt["recipe"] = {"path": str(args.recipe), "sha256": sha256_file(args.recipe),
                             "content_hash": recipe.get("content_hash")}
    result = convert(args.checkpoint, args.output or Path("/dev/null"), first_layer, layer_count, receipt,
                     args.dry_run, args.tp_degree, args.tp_rank, ffn_format=args.ffn_format)
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
