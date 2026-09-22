#!/usr/bin/env python3
"""Convert a MiMo 2.6 checkpoint (pro-rl / flash-rl) into mimo26 stage packs.

Setup-time code, never the serving path. The pack moves bytes as shipped -
MXFP4 expert payloads (U8, two e2m1 per byte) with their E8M0 group-32 scale
planes, fp8 e4m3 spine weights with F32 scale_inv [128,128] grids, and BF16
o_proj/embeddings/lm_head/norms/sink/router - NOTHING is recomputed or
requantized (the operator quality law). The v1 wire is the Q6SP/26I2Q
discipline shared with the minimax/qwen38_27b packers (120-byte header,
56-byte 6I4Q entries, 256-byte payload alignment) via tools/spark_pack_common.

Scope (model-families/mimo26/FACTS.md): the TEXT TOWER only. The MTP head,
vision/audio towers, speech embeddings and the dflash draft tree are never
referenced - they are documented with byte ranges in the census, not stripped
from the checkpoint.

Sharding (per the M1 topology decision; tools/mimo26_param_budget.py):
  pro   TP8: q row-sliced by head groups (16 of 128 heads), kv REPLICATED
             (192/128-row kv sections cannot cut the fp8 grid at head
             boundaries; whole-section replication costs ~16 MB/layer and
             keeps every scale grid whole), o_proj col-sliced by head-group
             v-dims, embed/lm_head vocab-row-sliced, router + norms +
             sink-slice replicated or head-sliced, 48 of 384 experts per
             layer per rank.
  flash TP4: 16 of 64 q heads, kv replicated, 64 of 256 experts, vocab/4.
  Expert slabs are per (layer, kind), local experts concatenated expert-major
  so one expert's rows are a base + e*extent window (a lazy-expert manifest
  can address single experts inside the slab).

The fused qkv_proj source tensor is emitted as THREE entries (q, k, v) whose
payload rows and scale rows come from the fused tensor's sections; the source
grid's tail padding rows (pro 212->216) are not data, and verify proves every
byte range by re-deriving it from the checkpoint.

Emission is staged and resumable past queue TTLs: `--emit` writes one payload/
scale file per directory entry into the stage dir (skipping files already
present at the exact expected size) and appends a journal line per record;
`--assemble` frames the header + directory, streams staged bytes into
<out>.partial, fsyncs, renames, and writes the .sha256 sidecar (sha256sum
format) plus a receipt; `--verify` re-derives the plan and byte-compares every
payload and scale plane against the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass, field
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

MAGIC = 0x5036324D  # 'M26P' little endian
FORMAT_VERSION = 1
HEADER_BYTES = 120
ENTRY_BYTES = 56
PAYLOAD_ALIGNMENT = 256
GLOBAL_LAYER = 0xFFFFFFFF

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_E4M3_F32B128 = 4
WEIGHT_MXFP4_E2M1_E8M0G32 = 9  # mirrors SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1_E8M0G32

HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES

FP8_BLOCK = 128
MXFP4_GROUP = 32
BF16_BYTES = 2
F32_BYTES = 4
CHUNK_BYTES = 512 * 1024

# span modes: how one contiguous output plane is produced from the checkpoint
SPAN_DENSE = "dense"        # row window of a [rows, cols] tensor, all columns
SPAN_RECT = "rect"          # row/col window of a [rows, cols] tensor
SPAN_SCALE_ROWS = "scale"   # row window (optionally col window) of a grid
SPAN_MX = "mx"              # one expert's packed payload or its scale plane

# kinds 0..5 mirror the shared SparkStagePackCommonTensorKind; the
# family-local kinds start at 22 (the qwen4_flash convention) so they never
# collide with the shared GDN/attention block or another family's extension.
(KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD, KIND_ATTENTION_NORM,
 KIND_MLP_NORM, KIND_MOE_GATE) = range(6)
(KIND_SINK_BIAS, KIND_MOE_GATE_BIAS,
 KIND_Q, KIND_K, KIND_V, KIND_O_PROJ,
 KIND_DENSE_MLP_GATE, KIND_DENSE_MLP_UP, KIND_DENSE_MLP_DOWN,
 KIND_EXPERT_GATE, KIND_EXPERT_UP, KIND_EXPERT_DOWN) = range(22, 34)


@dataclass
class Span:
    mode: str
    name: str            # payload tensor name (SPAN_MX: the expert's weight)
    dtype: str
    row0: int = 0
    rows: int = 0
    full_columns: int = 0
    col_base: int = 0
    columns: int = 0
    scale_name: str = ""
    scale_row0: int = 0
    scale_rows: int = 0
    scale_col_base: int = 0
    scale_columns: int = 0


@dataclass
class Record:
    kind: int
    layer: int
    weight_format: int
    rows: int
    columns: int
    name: str
    spans: list = field(default_factory=list)
    payload_bytes: int = 0
    scale_bytes: int = 0


ARMS = {
    "pro": dict(
        hidden=6144, layers=70, heads=128, head_dim=192, v_head_dim=128,
        kv_full=8, kv_swa=8, vocab=152576, dense_inter=16384,
        experts=384, experts_per_token=8, expert_inter=2048,
        swa_window=128, default_tp=8,
    ),
    "flash": dict(
        hidden=4096, layers=48, heads=64, head_dim=192, v_head_dim=128,
        kv_full=4, kv_swa=8, vocab=152576, dense_inter=16384,
        experts=256, experts_per_token=8, expert_inter=2048,
        swa_window=128, default_tp=4,
    ),
}


def arm_geometry(arm: str) -> dict:
    if arm not in ARMS:
        raise PackFailure(f"unknown arm {arm!r} (expected one of {sorted(ARMS)})")
    return ARMS[arm]


def layer_kind(config: dict, layer: int) -> str:
    pattern = config.get("hybrid_layer_pattern")
    if pattern is None or len(pattern) <= layer:
        raise PackFailure("config hybrid_layer_pattern missing the layer")
    return "full" if pattern[layer] == 0 else "swa"


def layer_is_moe(config: dict, layer: int) -> bool:
    freq = config.get("moe_layer_freq")
    if freq is None or len(freq) <= layer:
        raise PackFailure("config moe_layer_freq missing the layer")
    return bool(freq[layer])


def quant_bytes(rows: int, columns: int, weight_format: int) -> tuple:
    if weight_format == WEIGHT_BF16:
        return rows * columns * BF16_BYTES, 0
    if weight_format == WEIGHT_F32:
        return rows * columns * F32_BYTES, 0
    if weight_format == WEIGHT_FP8_E4M3_F32B128:
        if rows % FP8_BLOCK or columns % FP8_BLOCK:
            raise PackFailure(
                f"fp8 slice [{rows}, {columns}] is not whole [128,128] blocks")
        return rows * columns, (rows // FP8_BLOCK) * (columns // FP8_BLOCK) * F32_BYTES
    if weight_format == WEIGHT_MXFP4_E2M1_E8M0G32:
        if rows % MXFP4_GROUP or columns % MXFP4_GROUP:
            raise PackFailure(
                f"mxfp4 slab [{rows}, {columns}] is not whole groups of {MXFP4_GROUP}")
        return rows * columns // 2, rows * columns // MXFP4_GROUP
    raise PackFailure(f"weight format {weight_format} has no byte rule")


def block_window(row0: int, rows: int) -> tuple:
    """Scale-grid row window for a payload row window; fails closed unless
    the payload window lands on fp8 block boundaries."""
    if row0 % FP8_BLOCK or rows % FP8_BLOCK:
        raise PackFailure(
            f"rows [{row0}, +{rows}) do not land on {FP8_BLOCK}-row blocks")
    return row0 // FP8_BLOCK, rows // FP8_BLOCK


def fused_scale_name(fused_weight_name: str) -> str:
    """The fused qkv scale tensor is named off the module
    (qkv_proj.weight_scale_inv), not off the weight tensor."""
    if not fused_weight_name.endswith(".weight"):
        raise PackFailure(f"not a weight tensor name: {fused_weight_name}")
    return fused_weight_name[:-len(".weight")] + ".weight_scale_inv"


def qkv_sections(arm: str, config: dict, layer: int) -> dict:
    g = arm_geometry(arm)
    kind = layer_kind(config, layer)
    kv = g["kv_full"] if kind == "full" else g["kv_swa"]
    q0 = 0
    q_rows = g["heads"] * g["head_dim"]
    k0 = q0 + q_rows
    k_rows = kv * g["head_dim"]
    v0 = k0 + k_rows
    v_rows = kv * g["v_head_dim"]
    return dict(kind=kind, q0=q0, q_rows=q_rows, k0=k0, k_rows=k_rows,
                v0=v0, v_rows=v_rows, total=v0 + v_rows)


def check_source(arm: str, source: SafetensorsSource) -> dict:
    g = arm_geometry(arm)
    source.check_config({
        "model_type": "mimo_v2",
        "hidden_size": g["hidden"],
        "num_hidden_layers": g["layers"],
        "num_attention_heads": g["heads"],
        "head_dim": g["head_dim"],
        "v_head_dim": g["v_head_dim"],
        "num_key_value_heads": g["kv_full"],
        "swa_num_key_value_heads": g["kv_swa"],
        "vocab_size": g["vocab"],
        "intermediate_size": g["dense_inter"],
        "n_routed_experts": g["experts"],
        "num_experts_per_tok": g["experts_per_token"],
        "moe_intermediate_size": g["expert_inter"],
        "tie_word_embeddings": False,
    })
    quant = source.config.get("quantization_config") or {}
    if quant.get("fmt") != "e4m3":
        raise PackFailure("source quantization_config.fmt is not e4m3")
    ignored = quant.get("ignored_layers") or []
    for layer in range(g["layers"]):
        # the release lists module paths without the .weight suffix
        module = f"model.layers.{layer}.self_attn.o_proj"
        if module not in ignored:
            raise PackFailure(f"{module} is not on the fp8 ignore list; the "
                              "BF16-o_proj pin does not hold for this release")
    return g


def check_shapes(arm: str, source: SafetensorsSource) -> None:
    g = arm_geometry(arm)
    source.check_shape("model.embed_tokens.weight", g["vocab"], g["hidden"])
    source.check_shape("model.norm.weight", 1, g["hidden"])
    source.check_shape("lm_head.weight", g["vocab"], g["hidden"])
    grid_columns = g["hidden"] // FP8_BLOCK
    for layer in range(g["layers"]):
        sec = qkv_sections(arm, source.config, layer)
        fused = f"model.layers.{layer}.self_attn.qkv_proj.weight"
        source.check_shape(fused, sec["total"], g["hidden"], dtype="F8_E4M3")
        _, smeta, _ = source.resolve(fused_scale_name(fused))
        if smeta["dtype"] != "F32" or smeta["shape"][1] != grid_columns \
                or smeta["shape"][0] < sec["total"] // FP8_BLOCK:
            raise PackFailure(
                f"{fused}.weight_scale_inv: {smeta['dtype']} {smeta['shape']}, "
                f"expected F32 grid >= [{sec['total'] // FP8_BLOCK}, {grid_columns}] "
                "(upstream row padding is allowed; it is not data)")
        source.check_shape(f"model.layers.{layer}.self_attn.o_proj.weight",
                           g["hidden"], g["heads"] * g["v_head_dim"])
        source.check_shape(f"model.layers.{layer}.input_layernorm.weight", 1, g["hidden"])
        source.check_shape(f"model.layers.{layer}.post_attention_layernorm.weight",
                           1, g["hidden"])
        if layer_kind(source.config, layer) == "swa":
            source.check_shape(f"model.layers.{layer}.self_attn.attention_sink_bias",
                               1, g["heads"])
        if layer_is_moe(source.config, layer):
            source.check_shape(f"model.layers.{layer}.mlp.gate.weight",
                               g["experts"], g["hidden"])
            source.check_shape(
                f"model.layers.{layer}.mlp.gate.e_score_correction_bias",
                1, g["experts"], dtype="F32")
            for expert in (0, g["experts"] - 1):
                for mat, rows, cols in (
                    ("gate_proj", g["expert_inter"], g["hidden"]),
                    ("up_proj", g["expert_inter"], g["hidden"]),
                    ("down_proj", g["hidden"], g["expert_inter"]),
                ):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{mat}.weight"
                    source.check_shape(name, rows, cols // 2, dtype="U8")
                    source.check_shape(name.replace(".weight", ".weight_scale"),
                                       rows, cols // MXFP4_GROUP, dtype="U8")
        else:
            for mat, rows, cols in (
                ("gate_proj", g["dense_inter"], g["hidden"]),
                ("up_proj", g["dense_inter"], g["hidden"]),
                ("down_proj", g["hidden"], g["dense_inter"]),
            ):
                name = f"model.layers.{layer}.mlp.{mat}.weight"
                source.check_shape(name, rows, cols, dtype="F8_E4M3")
                source.check_shape(name + "_scale_inv",
                                   rows // FP8_BLOCK, cols // FP8_BLOCK, dtype="F32")


def build_plan(arm: str, config: dict, tp_degree: int, tp_rank: int,
               layer_window=None) -> list:
    """Directory records for one rank. layer_window (first, count) restricts
    the per-layer records to a contiguous window - globals always included -
    so fleet emission fans out per stage-slice under the queue TTL while the
    staging files stay content-addressed and merge by union."""
    g = arm_geometry(arm)
    for name, dimension in (("heads", g["heads"]), ("vocab", g["vocab"]),
                            ("experts", g["experts"]),
                            ("dense_inter", g["dense_inter"]),
                            ("hidden", g["hidden"])):
        if dimension % tp_degree:
            raise PackFailure(f"{name} {dimension} not divisible by tp {tp_degree}")
    q_rows_rank = (g["heads"] // tp_degree) * g["head_dim"]
    if q_rows_rank % FP8_BLOCK:
        raise PackFailure("per-rank q rows are not whole fp8 blocks")
    records: list = []

    def add(kind, layer, fmt, rows, columns, name, spans):
        payload, scale = quant_bytes(rows, columns, fmt)
        records.append(Record(kind=kind, layer=layer, weight_format=fmt,
                              rows=rows, columns=columns, name=name, spans=spans,
                              payload_bytes=payload, scale_bytes=scale))

    vocab_rows = g["vocab"] // tp_degree
    add(KIND_EMBEDDING, GLOBAL_LAYER, WEIGHT_BF16, vocab_rows, g["hidden"],
        "model.embed_tokens.weight",
        [Span(SPAN_DENSE, "model.embed_tokens.weight", "BF16",
              tp_rank * vocab_rows, vocab_rows, g["hidden"])])
    add(KIND_FINAL_NORM, GLOBAL_LAYER, WEIGHT_BF16, 1, g["hidden"],
        "model.norm.weight",
        [Span(SPAN_DENSE, "model.norm.weight", "BF16", 0, 1, g["hidden"])])
    add(KIND_LM_HEAD, GLOBAL_LAYER, WEIGHT_BF16, vocab_rows, g["hidden"],
        "lm_head.weight",
        [Span(SPAN_DENSE, "lm_head.weight", "BF16",
              tp_rank * vocab_rows, vocab_rows, g["hidden"])])

    q_heads = g["heads"] // tp_degree
    o_cols = q_heads * g["v_head_dim"]
    local_experts = g["experts"] // tp_degree
    expert_base = tp_rank * local_experts
    dense_rows = g["dense_inter"] // tp_degree
    dense_cols = g["dense_inter"] // tp_degree
    if layer_window is not None:
        first, count = layer_window
        if first < 0 or count < 1 or first + count > g["layers"]:
            raise PackFailure(f"layer window [{first}, +{count}) out of range")
        layers = range(first, first + count)
    else:
        layers = range(g["layers"])
    for layer in layers:
        sec = qkv_sections(arm, config, layer)
        fused = f"model.layers.{layer}.self_attn.qkv_proj.weight"
        fused_scale = fused_scale_name(fused)
        grid_columns = g["hidden"] // FP8_BLOCK
        # q: this rank's head-group rows inside the fused q section
        q_row0 = sec["q0"] + tp_rank * q_heads * g["head_dim"]
        s0, sc = block_window(q_row0, q_rows_rank)
        add(KIND_Q, layer, WEIGHT_FP8_E4M3_F32B128, q_rows_rank, g["hidden"],
            fused + "#q",
            [Span(SPAN_DENSE, fused, "F8_E4M3", q_row0, q_rows_rank, g["hidden"],
                  scale_name=fused_scale, scale_row0=s0, scale_rows=sc,
                  scale_columns=grid_columns)])
        # k/v: whole sections replicated (kv-head granules cannot cut the grid)
        for kind_code, sec0, sec_rows, tag in (
                (KIND_K, sec["k0"], sec["k_rows"], "k"),
                (KIND_V, sec["v0"], sec["v_rows"], "v")):
            gs0, gsc = block_window(sec0, sec_rows)
            add(kind_code, layer, WEIGHT_FP8_E4M3_F32B128, sec_rows, g["hidden"],
                fused + "#" + tag,
                [Span(SPAN_DENSE, fused, "F8_E4M3", sec0, sec_rows, g["hidden"],
                      scale_name=fused_scale, scale_row0=gs0, scale_rows=gsc,
                      scale_columns=grid_columns)])
        o_name = f"model.layers.{layer}.self_attn.o_proj.weight"
        add(KIND_O_PROJ, layer, WEIGHT_BF16, g["hidden"], o_cols, o_name,
            [Span(SPAN_RECT, o_name, "BF16", 0, g["hidden"],
                  g["heads"] * g["v_head_dim"], tp_rank * o_cols, o_cols)])
        add(KIND_ATTENTION_NORM, layer, WEIGHT_BF16, 1, g["hidden"],
            f"model.layers.{layer}.input_layernorm.weight",
            [Span(SPAN_DENSE, f"model.layers.{layer}.input_layernorm.weight",
                  "BF16", 0, 1, g["hidden"])])
        add(KIND_MLP_NORM, layer, WEIGHT_BF16, 1, g["hidden"],
            f"model.layers.{layer}.post_attention_layernorm.weight",
            [Span(SPAN_DENSE, f"model.layers.{layer}.post_attention_layernorm.weight",
                  "BF16", 0, 1, g["hidden"])])
        if layer_kind(config, layer) == "swa":
            sink = f"model.layers.{layer}.self_attn.attention_sink_bias"
            add(KIND_SINK_BIAS, layer, WEIGHT_BF16, 1, q_heads, sink,
                [Span(SPAN_RECT, sink, "BF16", 0, 1, g["heads"],
                      tp_rank * q_heads, q_heads)])
        if layer_is_moe(config, layer):
            gate = f"model.layers.{layer}.mlp.gate.weight"
            add(KIND_MOE_GATE, layer, WEIGHT_BF16, g["experts"], g["hidden"], gate,
                [Span(SPAN_DENSE, gate, "BF16", 0, g["experts"], g["hidden"])])
            bias = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
            add(KIND_MOE_GATE_BIAS, layer, WEIGHT_F32, 1, g["experts"], bias,
                [Span(SPAN_DENSE, bias, "F32", 0, 1, g["experts"])])
            for kind_code, mat, rows_each, cols in (
                    (KIND_EXPERT_GATE, "gate_proj", g["expert_inter"], g["hidden"]),
                    (KIND_EXPERT_UP, "up_proj", g["expert_inter"], g["hidden"]),
                    (KIND_EXPERT_DOWN, "down_proj", g["hidden"], g["expert_inter"])):
                spans = []
                for expert in range(expert_base, expert_base + local_experts):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{mat}.weight"
                    spans.append(Span(
                        SPAN_MX, name, "U8", 0, rows_each, cols,
                        scale_name=name.replace(".weight", ".weight_scale")))
                add(kind_code, layer, WEIGHT_MXFP4_E2M1_E8M0G32,
                    local_experts * rows_each, cols,
                    f"model.layers.{layer}.mlp.experts.[{expert_base},"
                    f"{expert_base + local_experts}).{mat}", spans)
        else:
            for kind_code, mat in ((KIND_DENSE_MLP_GATE, "gate_proj"),
                                   (KIND_DENSE_MLP_UP, "up_proj")):
                name = f"model.layers.{layer}.mlp.{mat}.weight"
                s0, sc = block_window(tp_rank * dense_rows, dense_rows)
                add(kind_code, layer, WEIGHT_FP8_E4M3_F32B128,
                    dense_rows, g["dense_inter"], name,
                    [Span(SPAN_DENSE, name, "F8_E4M3", tp_rank * dense_rows,
                          dense_rows, g["dense_inter"],
                          scale_name=name + "_scale_inv", scale_row0=s0,
                          scale_rows=sc, scale_columns=g["dense_inter"] // FP8_BLOCK)])
            name = f"model.layers.{layer}.mlp.down_proj.weight"
            c0, cc = block_window(tp_rank * dense_cols, dense_cols)
            add(KIND_DENSE_MLP_DOWN, layer, WEIGHT_FP8_E4M3_F32B128,
                g["hidden"], dense_cols, name,
                [Span(SPAN_RECT, name, "F8_E4M3", 0, g["hidden"], g["dense_inter"],
                      tp_rank * dense_cols, dense_cols,
                      scale_name=name + "_scale_inv", scale_row0=0,
                      scale_rows=g["hidden"] // FP8_BLOCK,
                      scale_col_base=c0, scale_columns=cc)])
    return records


def element_bytes(dtype: str) -> int:
    return {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}[dtype]


class SourceReader:
    """Streams byte ranges out of the checkpoint; shard handles stay open."""

    def __init__(self, source: SafetensorsSource) -> None:
        self.source = source
        self.files: dict = {}

    def _file(self, shard: str):
        file = self.files.get(shard)
        if file is None or file.closed:
            file = open(self.source.root / shard, "rb")
            self.files[shard] = file
        return file

    def _resolve_base(self, name: str) -> tuple:
        shard, meta, base = self.source.resolve(name)
        return self._file(shard), base, meta

    def copy_dense(self, span: Span, out) -> None:
        file, base, _ = self._resolve_base(span.name)
        row_bytes = span.full_columns * element_bytes(span.dtype)
        file.seek(base + span.row0 * row_bytes)
        remaining = span.rows * row_bytes
        while remaining > 0:
            step = min(remaining, CHUNK_BYTES)
            chunk = file.read(step)
            if len(chunk) != step:
                raise PackFailure(f"{span.name}: short dense read")
            out.write(chunk)
            remaining -= step

    def copy_rect(self, span: Span, out) -> None:
        file, base, _ = self._resolve_base(span.name)
        element = element_bytes(span.dtype)
        row_bytes = span.full_columns * element
        step_rows = max(1, CHUNK_BYTES // max(1, span.columns * element))
        row = 0
        while row < span.rows:
            block = min(step_rows, span.rows - row)
            file.seek(base + (span.row0 + row) * row_bytes + span.col_base * element)
            remaining = block * span.columns * element
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                chunk = file.read(step)
                if len(chunk) != step:
                    raise PackFailure(f"{span.name}: short rect read")
                out.write(chunk)
                remaining -= step
            row += block

    def copy_scale(self, span: Span, out) -> None:
        if span.scale_name not in self.source.weight_map:
            raise PackFailure(f"scale tensor missing: {span.scale_name}")
        file, base, meta = self._resolve_base(span.scale_name)
        grid_columns = meta["shape"][1]
        columns = span.scale_columns or grid_columns
        element = element_bytes("F32")
        row_bytes = grid_columns * element
        step_rows = max(1, CHUNK_BYTES // max(1, columns * element))
        row = 0
        while row < span.scale_rows:
            block = min(step_rows, span.scale_rows - row)
            file.seek(base + (span.scale_row0 + row) * row_bytes
                      + span.scale_col_base * element)
            remaining = block * columns * element
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                chunk = file.read(step)
                if len(chunk) != step:
                    raise PackFailure(f"{span.scale_name}: short scale read")
                out.write(chunk)
                remaining -= step
            row += block

    def produce(self, span: Span, record: Record, out, payload: bool) -> None:
        if span.mode == SPAN_MX:
            if payload:
                packed = Span(SPAN_DENSE, span.name, "U8", 0, span.rows,
                              span.full_columns // 2)
                self.copy_dense(packed, out)
            else:
                scale_span = Span(SPAN_DENSE, span.scale_name, "U8", 0, span.rows,
                                  span.full_columns // MXFP4_GROUP)
                self.copy_dense(scale_span, out)
            return
        if payload:
            if span.mode == SPAN_DENSE:
                self.copy_dense(span, out)
            elif span.mode == SPAN_RECT:
                self.copy_rect(span, out)
            else:
                raise PackFailure(f"payload mode {span.mode} unknown")
            return
        if span.scale_name:
            self.copy_scale(span, out)
            return
        if record.scale_bytes:
            raise PackFailure(f"record {record.name} has scale bytes but span "
                              f"{span.name} names no scale tensor")


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stage_name(record: Record, plane: str) -> str:
    layer = "g" if record.layer == GLOBAL_LAYER else f"{record.layer:03d}"
    return f"r{layer}_{record.kind:02d}.{plane}"


def emit_record(reader: SourceReader, record: Record, stage_dir: Path):
    """Stage one record's payload/scale files; resume skips exact-size files."""
    payload_path = stage_dir / stage_name(record, "payload")
    scale_path = stage_dir / stage_name(record, "scale")
    if not (payload_path.exists() and payload_path.stat().st_size == record.payload_bytes):
        with open(payload_path, "wb") as out:
            for span in record.spans:
                reader.produce(span, record, out, payload=True)
            out.flush()
            os.fsync(out.fileno())
    if record.scale_bytes and not (scale_path.exists()
                                   and scale_path.stat().st_size == record.scale_bytes):
        with open(scale_path, "wb") as out:
            for span in record.spans:
                reader.produce(span, record, out, payload=False)
            out.flush()
            os.fsync(out.fileno())


def header_fields(arm: str, records: list, directory_offset: int,
                  file_bytes: int, tp_degree: int, tp_rank: int):
    """All 26 u32 slots carry SparkStagePackHeaderCommon semantics (the
    module's layout proof asserts field-for-field offset equality); the mimo
    hybrid pattern is irregular so attention_period and the gdn block are
    zero and the layer kinds bind via the family geometry tables. TP identity
    is environment-driven and receipt-pinned (the qwen4_flash convention),
    not a pack header field."""
    g = arm_geometry(arm)
    return (
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(records),
        g["hidden"], g["layers"], 0, g["layers"],
        0, 0,
        0, 0, 0, 0, 0,
        g["heads"], g["kv_full"], g["head_dim"], 64,
        g["experts"], g["experts_per_token"], g["expert_inter"],
        g["vocab"], MXFP4_GROUP,
        0,
        directory_offset, file_bytes,
    )


def plan_layout(records: list) -> tuple:
    """256-aligned (payload, scale) offsets per record plus total file bytes."""
    layout = []
    cursor = HEADER_BYTES + ENTRY_BYTES * len(records)
    for record in records:
        payload_offset = align_up(cursor, PAYLOAD_ALIGNMENT)
        cursor = payload_offset + record.payload_bytes
        scale_offset = 0
        if record.scale_bytes:
            scale_offset = cursor
            cursor += record.scale_bytes
        layout.append((payload_offset, scale_offset))
    return layout, cursor


def parse_window(text):
    if text is None:
        return None
    try:
        first, count = (int(part) for part in text.split(":"))
    except ValueError:
        raise PackFailure(f"--layer-window expects FIRST:COUNT, got {text!r}")
    return (first, count)


def preflight(source: SafetensorsSource, stage_dir: Path) -> dict:
    """Emission pre-flight: measure the metadata plane (one stat per source
    shard - a stalled MDS band shows up here as a slow walk, not as a silent
    crawl mid-emission) and pin the carving tool's own bytes against the
    archive ledger so the carve is provenance-clean."""
    import time as _time
    shards = sorted(set(source.weight_map.values()))
    t0 = _time.perf_counter()
    missing = []
    for shard in shards:
        if not (source.root / shard).is_file():
            missing.append(shard)
    walk_seconds = _time.perf_counter() - t0
    if missing:
        raise PackFailure(f"preflight: {len(missing)} source shards missing, "
                          f"first {missing[0]}")
    ledger = (Path(__file__).resolve().parents[1] / "SHA256SUMS")
    tool_sha = None
    if ledger.is_file():
        want = next((line.split()[0] for line in ledger.read_text().splitlines()
                     if line.endswith("  tools/mimo26_stagepack.py")), None)
        if want:
            tool_sha = sha256_file(Path(__file__).resolve())
            if tool_sha != want:
                raise PackFailure("preflight: packer bytes do not match the "
                                  "archive SHA256SUMS pin")
    print(f"preflight: {len(shards)} shards stat-walked in {walk_seconds:.2f}s"
          + (f", tool sha pinned ({tool_sha[:12]}...)" if tool_sha else ""))
    if walk_seconds > 30.0:
        raise PackFailure(f"preflight: metadata walk took {walk_seconds:.1f}s - "
                          "the MDS band is stalled; refusing to emit")
    return {"shards": len(shards), "stat_walk_seconds": round(walk_seconds, 3),
            "tool_sha256_pinned": bool(tool_sha)}


def emit_file_major(source: SafetensorsSource, records: list, stage_dir: Path,
                    reader: SourceReader) -> int:
    """Default emission shape: every span grouped by its SOURCE SHARD, one
    open per shard, spans in file-offset order, bytes appended to the
    content-addressed staging files. 16 sequential streams replace ~2900
    per-tensor MDS lookups per rank (the file-major ruling)."""
    from collections import defaultdict
    by_shard = defaultdict(list)
    for record in records:
        for plane in (True, False):
            for span in record.spans:
                for name in (payload_name(span, plane),):
                    shard = source.weight_map.get(name)
                    if shard is None:
                        raise PackFailure(f"span source not in index: {name}")
                    by_shard[shard].append((record, span, plane))
    done = 0
    outs = {}
    plane_state = {}  # path -> "skip" | "write"; decided once per plane
    try:
        for shard in sorted(by_shard):
            items = by_shard[shard]
            file = reader._file(shard)
            for record, span, plane in items:
                path = stage_dir / (stage_name(record, "payload" if plane else "scale"))
                want = record.payload_bytes if plane else record.scale_bytes
                state = plane_state.get(path)
                if state is None:
                    if path.exists() and path.stat().st_size == want:
                        plane_state[path] = "skip"  # complete planes resume
                        state = "skip"
                    else:
                        # a PARTIAL plane cannot be resumed across emission
                        # layouts (span order differs); discard and rewrite
                        if path.exists():
                            path.unlink()
                        plane_state[path] = "write"
                        state = "write"
                        done += want
                if state == "skip":
                    continue
                out = outs.get(path)
                if out is None:
                    out = open(path, "wb")
                    outs[path] = out
                reader.produce(span, record, out, payload=plane)
    finally:
        for out in outs.values():
            out.flush()
            os.fsync(out.fileno())
            out.close()
    # any record that staged nothing in this pass (already complete) counts too
    return done


def payload_name(span, plane):
    if not plane and span.scale_name:
        return span.scale_name
    return span.name


def do_emit(args) -> int:
    source = SafetensorsSource(Path(args.checkpoint))
    check_source(args.arm, source)
    check_shapes(args.arm, source)
    preflight(source, Path(args.stage_dir or (str(args.out) + ".stage")))
    records = build_plan(args.arm, source.config, args.tp, args.rank,
                         parse_window(getattr(args, "layer_window", None)))
    stage_dir = Path(args.stage_dir or (str(args.out) + ".stage"))
    stage_dir.mkdir(parents=True, exist_ok=True)
    reader = SourceReader(source)
    journal_path = stage_dir / "journal.jsonl"
    done_bytes = emit_file_major(source, records, stage_dir, reader)
    with open(journal_path, "a", encoding="utf-8") as journal:
        for index, record in enumerate(records):
            journal.write(json.dumps({
                "index": index, "kind": record.kind, "layer": record.layer,
                "name": record.name, "rows": record.rows, "columns": record.columns,
                "weight_format": record.weight_format,
                "payload_bytes": record.payload_bytes,
                "scale_bytes": record.scale_bytes}) + "\n")
    print(f"emit: {len(records)} records, {done_bytes} new bytes staged under {stage_dir} (file-major)")
    return 0


def do_assemble(args) -> int:
    source = SafetensorsSource(Path(args.checkpoint))
    check_source(args.arm, source)
    records = build_plan(args.arm, source.config, args.tp, args.rank)
    stage_dir = Path(args.stage_dir or (str(args.out) + ".stage"))
    layout, file_bytes = plan_layout(records)
    directory_offset = HEADER_BYTES
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".partial")
    with open(partial, "wb") as out:
        out.write(HEADER_STRUCT.pack(*header_fields(
            args.arm, records, directory_offset, file_bytes, args.tp, args.rank)))
        for record, (payload_offset, scale_offset) in zip(records, layout):
            layer = 0xFFFFFFFF if record.layer == GLOBAL_LAYER else record.layer
            out.write(ENTRY_STRUCT.pack(record.kind, layer, record.weight_format,
                                        record.rows, record.columns, 0,
                                        payload_offset, record.payload_bytes,
                                        scale_offset, record.scale_bytes))
        for index, (record, (payload_offset, scale_offset)) in enumerate(
                zip(records, layout)):
            payload_path = stage_dir / stage_name(record, "payload")
            if not payload_path.exists() or \
                    payload_path.stat().st_size != record.payload_bytes:
                raise PackFailure(f"staged payload missing/wrong size: {payload_path}")
            out.seek(payload_offset)
            with open(payload_path, "rb") as f:
                remaining = record.payload_bytes
                while remaining > 0:
                    chunk = f.read(min(remaining, CHUNK_BYTES))
                    if not chunk:
                        raise PackFailure(f"short staged payload {payload_path}")
                    out.write(chunk)
                    remaining -= len(chunk)
            if record.scale_bytes:
                scale_path = stage_dir / stage_name(record, "scale")
                if not scale_path.exists() or \
                        scale_path.stat().st_size != record.scale_bytes:
                    raise PackFailure(f"staged scale missing/wrong size: {scale_path}")
                out.seek(scale_offset)
                with open(scale_path, "rb") as f:
                    remaining = record.scale_bytes
                    while remaining > 0:
                        chunk = f.read(min(remaining, CHUNK_BYTES))
                        if not chunk:
                            raise PackFailure(f"short staged scale {scale_path}")
                        out.write(chunk)
                        remaining -= len(chunk)
        if out.tell() != file_bytes:
            raise PackFailure(f"assembled {out.tell()} bytes, planned {file_bytes}")
        out.flush()
        os.fsync(out.fileno())
    os.replace(partial, out_path)
    digest = sha256_file(out_path)
    sha_path = out_path.with_name(out_path.name + ".sha256")
    sha_path.write_text(f"{digest}  {out_path.name}\n", encoding="utf-8")
    receipt = {
        "arm": args.arm,
        "topology": f"tp{args.tp}",
        "rank": args.rank,
        "file": str(out_path),
        "file_bytes": file_bytes,
        "tensor_count": len(records),
        "sha256": digest,
        "source_checkpoint": str(Path(args.checkpoint)),
        "source_config_sha256": source.config_sha256,
        "source_index_sha256": source.index_sha256,
        "weight_formats": sorted({r.weight_format for r in records}),
        "record_kinds": len({r.kind for r in records}),
    }
    write_receipt(receipt, out_path)
    print(f"assemble: {out_path} ({file_bytes} bytes, {len(records)} tensors, "
          f"sha256 {digest[:16]}...)")
    return 0


def do_verify(args) -> int:
    source = SafetensorsSource(Path(args.checkpoint))
    check_source(args.arm, source)
    records = build_plan(args.arm, source.config, args.tp, args.rank)
    reader = SourceReader(source)
    pack_path = Path(args.out)
    with open(pack_path, "rb") as pack:
        raw = pack.read(HEADER_BYTES)
        if len(raw) != HEADER_BYTES:
            raise PackFailure("short pack header")
        header = HEADER_STRUCT.unpack(raw)
        if header[0] != MAGIC or header[1] != FORMAT_VERSION:
            raise PackFailure("pack magic/version mismatch")
        tensor_count = header[4]
        if tensor_count != len(records):
            raise PackFailure(
                f"pack has {tensor_count} tensors, plan has {len(records)}")
        pack.seek(HEADER_BYTES)
        entries = [ENTRY_STRUCT.unpack(pack.read(ENTRY_BYTES))
                   for _ in range(tensor_count)]
        for index, (record, entry) in enumerate(zip(records, entries)):
            kind, layer, fmt, rows, columns, _res, payload_offset, payload_bytes, \
                scale_offset, scale_bytes = entry
            want_layer = 0xFFFFFFFF if record.layer == GLOBAL_LAYER else record.layer
            if (kind, layer, fmt, rows, columns, payload_bytes, scale_bytes) != (
                    record.kind, want_layer, record.weight_format,
                    record.rows, record.columns,
                    record.payload_bytes, record.scale_bytes):
                raise PackFailure(
                    "entry %d directory mismatch: pack %s plan %s" % (
                        index,
                        (kind, layer, fmt, rows, columns, payload_bytes, scale_bytes),
                        (record.kind, want_layer, record.weight_format, record.rows,
                         record.columns, record.payload_bytes, record.scale_bytes)))
            if scale_bytes and scale_offset < payload_offset + payload_bytes:
                raise PackFailure(f"entry {index} scale plane overlaps payload")
        for index, (record, entry) in enumerate(zip(records, entries)):
            _k, _l, _f, _r, _c, _res, payload_offset, _pb, scale_offset, _sb = entry
            pack.seek(payload_offset)
            sink = _CompareSink(pack, f"entry {index} ({record.name}) payload")
            for span in record.spans:
                reader.produce(span, record, sink, payload=True)
            sink.expect(record.payload_bytes)
            if record.scale_bytes:
                pack.seek(scale_offset)
                sink = _CompareSink(pack, f"entry {index} ({record.name}) scale")
                for span in record.spans:
                    reader.produce(span, record, sink, payload=False)
                sink.expect(record.scale_bytes)
    receipt = json.loads(
        pack_path.with_name(pack_path.name + ".receipt.json").read_text())
    digest = sha256_file(pack_path)
    if receipt["sha256"] != digest:
        raise PackFailure("receipt sha256 mismatch")
    if receipt["file_bytes"] != pack_path.stat().st_size:
        raise PackFailure("receipt file_bytes mismatch")
    print(f"verify: {pack_path} byte-exact against the checkpoint "
          f"({len(records)} tensors, sha256 {digest[:16]}...)")
    return 0


class _CompareSink:
    """Streaming verifier sink: every produced chunk is compared in place
    against the pack file's current position (no plane is ever buffered)."""

    def __init__(self, pack, label: str) -> None:
        self.pack = pack
        self.label = label
        self.count = 0

    def write(self, chunk: bytes) -> int:
        got = self.pack.read(len(chunk))
        if got != chunk:
            raise PackFailure(f"{self.label} differs from source at byte {self.count}")
        self.count += len(chunk)
        return len(chunk)

    def expect(self, total: int) -> None:
        if self.count != total:
            raise PackFailure(f"{self.label}: plan derives {self.count} bytes, "
                              f"directory says {total}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-dir")
    ap.add_argument("--layer-window", default=None,
                    help="FIRST:COUNT - emit globals plus this layer window only")
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--emit", action="store_true")
    sub.add_argument("--assemble", action="store_true")
    sub.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.rank < 0 or args.rank >= args.tp:
        raise PackFailure(f"rank {args.rank} out of range for tp {args.tp}")
    if args.emit:
        return do_emit(args)
    if args.assemble:
        return do_assemble(args)
    return do_verify(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackFailure as failure:
        print(f"PACK FAILURE: {failure}", file=sys.stderr)
        raise SystemExit(1)
