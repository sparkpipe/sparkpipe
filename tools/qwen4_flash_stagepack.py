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
# v2: full-width hc norms, per-sublayer hc mixers, indexer, PLE, mixers.
FORMAT_VERSION = 2
HEADER_BYTES = 120
ENTRY_BYTES = 56
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE
PAYLOAD_ALIGNMENT = 256

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_F32B128 = 4
WEIGHT_FP8_E8M0B128 = 6
WEIGHT_I64 = 7

HIDDEN = 2560
HC_STREAMS = 4
HC_LOWRANK = 320
STREAM_WIDTH = HC_STREAMS * HIDDEN  # 10240
INDEXER_HEADS = 4
INDEXER_KV_HEADS = 1
INDEXER_HEAD_DIM = 128
PLE_LAYER = 1
PLE_NGRAM_HEADS = 16
PLE_NGRAM_HEAD_DIM = 160
PLE_NGRAM_ROWS = 320001536
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
(KIND_ATTN_HC_DOWN, KIND_ATTN_HC_UP, KIND_ATTN_HC_INJECT,
 KIND_MLP_HC_DOWN, KIND_MLP_HC_UP, KIND_MLP_HC_INJECT,
 KIND_INDEXER_QK, KIND_INDEXER_Q_NORM, KIND_INDEXER_K_NORM,
 KIND_MIXER_DOWN, KIND_MIXER_UP, KIND_MTP_MIXER_DOWN, KIND_MTP_MIXER_UP,
 KIND_PLE_KEY, KIND_PLE_VALUE, KIND_PLE_NORM_KEY, KIND_PLE_NORM_QUERY,
 KIND_PLE_NORM_CONV, KIND_PLE_CONV, KIND_PLE_MULTIPLIERS,
 KIND_PLE_HEAD_VOCABS, KIND_PLE_HEAD_OFFSETS,
 KIND_PLE_NGRAM) = range(32, 55)

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
        KIND_FINAL_NORM: (1, STREAM_WIDTH, WEIGHT_BF16),
        KIND_LM_HEAD: (VOCAB, HIDDEN, WEIGHT_BF16),
        KIND_ATTENTION_NORM: (1, STREAM_WIDTH, WEIGHT_BF16),
        KIND_MLP_NORM: (1, STREAM_WIDTH, WEIGHT_BF16),
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
        # v2: the norm slots carry the FULL 4-stream hc_norm widths.
        KIND_MTP_EMBED_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MTP_HIDDEN_NORM: (1, STREAM_WIDTH, WEIGHT_BF16),
        KIND_MTP_FINAL_NORM: (1, STREAM_WIDTH, WEIGHT_BF16),
        KIND_ATTN_HC_DOWN: (HC_LOWRANK, STREAM_WIDTH, WEIGHT_BF16),
        KIND_ATTN_HC_UP: (STREAM_WIDTH, HC_LOWRANK, WEIGHT_BF16),
        KIND_ATTN_HC_INJECT: (HC_STREAMS, STREAM_WIDTH, WEIGHT_BF16),
        KIND_MLP_HC_DOWN: (HC_LOWRANK, STREAM_WIDTH, WEIGHT_BF16),
        KIND_MLP_HC_UP: (STREAM_WIDTH, HC_LOWRANK, WEIGHT_BF16),
        KIND_MLP_HC_INJECT: (HC_STREAMS, STREAM_WIDTH, WEIGHT_BF16),
        KIND_INDEXER_QK: ((INDEXER_HEADS + INDEXER_KV_HEADS) * INDEXER_HEAD_DIM, HIDDEN, WEIGHT_BF16),
        KIND_INDEXER_Q_NORM: (1, INDEXER_HEAD_DIM, WEIGHT_BF16),
        KIND_INDEXER_K_NORM: (1, INDEXER_HEAD_DIM, WEIGHT_BF16),
        KIND_MIXER_DOWN: (HC_LOWRANK, STREAM_WIDTH, WEIGHT_BF16),
        KIND_MIXER_UP: (STREAM_WIDTH, HC_LOWRANK, WEIGHT_BF16),
        KIND_MTP_MIXER_DOWN: (HC_LOWRANK, STREAM_WIDTH, WEIGHT_BF16),
        KIND_MTP_MIXER_UP: (STREAM_WIDTH, HC_LOWRANK, WEIGHT_BF16),
        KIND_PLE_KEY: (STREAM_WIDTH, HIDDEN, WEIGHT_BF16),
        KIND_PLE_VALUE: (HIDDEN, HIDDEN, WEIGHT_BF16),
        KIND_PLE_NORM_KEY: (1, STREAM_WIDTH, WEIGHT_BF16),
        KIND_PLE_NORM_QUERY: (1, STREAM_WIDTH, WEIGHT_BF16),
        KIND_PLE_NORM_CONV: (1, STREAM_WIDTH, WEIGHT_BF16),
        KIND_PLE_CONV: (STREAM_WIDTH, 4, WEIGHT_BF16),
        KIND_PLE_MULTIPLIERS: (1, 3, WEIGHT_I64),
        KIND_PLE_HEAD_VOCABS: (1, PLE_NGRAM_HEADS, WEIGHT_I64),
        KIND_PLE_HEAD_OFFSETS: (1, PLE_NGRAM_HEADS, WEIGHT_I64),
        KIND_PLE_NGRAM: (PLE_NGRAM_ROWS, PLE_NGRAM_HEAD_DIM, WEIGHT_BF16),
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
        KIND_INDEXER_QK: "self_attn.indexer.index_qk_proj.weight",
        KIND_INDEXER_Q_NORM: "self_attn.indexer.q_layernorm.weight",
        KIND_INDEXER_K_NORM: "self_attn.indexer.k_layernorm.weight",
    }
    ple = {
        KIND_PLE_KEY: "ple.key_proj.weight",
        KIND_PLE_VALUE: "ple.value_proj.weight",
        KIND_PLE_NORM_KEY: "ple.norm_key.weight",
        KIND_PLE_NORM_QUERY: "ple.norm_query.weight",
        KIND_PLE_NORM_CONV: "ple.norm_conv.weight",
        KIND_PLE_CONV: "ple.conv1d.weight",
        KIND_PLE_MULTIPLIERS: "ple.ple_embedding.layer_multipliers",
        KIND_PLE_HEAD_VOCABS: "ple.ple_embedding.ngram_heads_vocab_sizes",
        KIND_PLE_HEAD_OFFSETS: "ple.ple_embedding.ngram_heads_offsets",
        # The n-gram table is 128 checkpoint shards; the name is a sentinel -
        # check_shape and the copier handle the shard span specially.
        KIND_PLE_NGRAM: "ple.ple_embedding.ngram_embedding",
    }
    every = {
        # v2 exact semantics (modeling_qwen4_exp.py): the hc_norm IS the
        # per-sublayer norm over the 4-stream residual, full [4H] width.
        KIND_ATTENTION_NORM: "attn_hyper_connection.hc_norm.weight",
        KIND_MLP_NORM: "mlp_hyper_connection.hc_norm.weight",
        KIND_ATTN_HC_DOWN: "attn_hyper_connection.input_mix_weight_down.weight",
        KIND_ATTN_HC_UP: "attn_hyper_connection.input_mix_weight_up.weight",
        KIND_ATTN_HC_INJECT: "attn_hyper_connection.block_inject_weight.weight",
        KIND_MLP_HC_DOWN: "mlp_hyper_connection.input_mix_weight_down.weight",
        KIND_MLP_HC_UP: "mlp_hyper_connection.input_mix_weight_up.weight",
        KIND_MLP_HC_INJECT: "mlp_hyper_connection.block_inject_weight.weight",
        KIND_MOE_GATE: "mlp.gate.weight",
        KIND_MOE_W1: "mlp.experts.gate_up_proj",
        KIND_MOE_W3: "mlp.experts.gate_up_proj",
        KIND_MOE_DOWN: "mlp.experts.down_proj",
        KIND_MOE_SHARED_GATE: "mlp.shared_expert.gate_proj.weight",
        KIND_MOE_SHARED_UP: "mlp.shared_expert.up_proj.weight",
        KIND_MOE_SHARED_DOWN: "mlp.shared_expert.down_proj.weight",
        KIND_MOE_SHARED_GATE_WEIGHT: "mlp.shared_expert_gate.weight",
    }
    for mapping in (every, gdn, attn, ple):
        if kind in mapping:
            return prefix + mapping[kind]
    raise PackFailure(f"kind {kind} is not a per-layer tensor")


GLOBAL_TENSORS = {
    KIND_EMBEDDING: "model.language_model.embed_tokens.weight",
    # v2: the readout is the hyper_connection_mixer itself - hc_norm plus
    # the low-rank mix pair (use_combine=False: mean-mix, no inject).
    KIND_FINAL_NORM: "model.language_model.hyper_connection_mixer.hc_norm.weight",
    KIND_MIXER_DOWN: "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    KIND_MIXER_UP: "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    KIND_LM_HEAD: "lm_head.weight",
    KIND_MTP_FC: MTP_PREFIX + "fc_embedding.weight+" + MTP_PREFIX + "fc_hidden.weight",
    KIND_MTP_EMBED_NORM: MTP_PREFIX + "pre_fc_norm_embedding.weight",
    KIND_MTP_HIDDEN_NORM: MTP_PREFIX + "pre_fc_norm_hidden.weight",
    KIND_MTP_FINAL_NORM: MTP_PREFIX + "hyper_connection_mixer.hc_norm.weight",
    KIND_MTP_MIXER_DOWN: MTP_PREFIX + "hyper_connection_mixer.input_mix_weight_down.weight",
    KIND_MTP_MIXER_UP: MTP_PREFIX + "hyper_connection_mixer.input_mix_weight_up.weight",
}

EVERY_LAYER_KINDS = (KIND_ATTENTION_NORM, KIND_MLP_NORM,
                     KIND_ATTN_HC_DOWN, KIND_ATTN_HC_UP, KIND_ATTN_HC_INJECT,
                     KIND_MLP_HC_DOWN, KIND_MLP_HC_UP, KIND_MLP_HC_INJECT,
                     KIND_MOE_GATE, KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN,
                     KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_UP,
                     KIND_MOE_SHARED_DOWN, KIND_MOE_SHARED_GATE_WEIGHT)
GDN_LAYER_KINDS = (KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY,
                   KIND_GDN_OUTPUT, KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG,
                   KIND_GDN_DT_BIAS, KIND_GDN_NORM)
ATTN_LAYER_KINDS = (KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE,
                    KIND_ATTN_OUTPUT, KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM,
                    KIND_INDEXER_QK, KIND_INDEXER_Q_NORM, KIND_INDEXER_K_NORM)
PLE_LAYER_KINDS = (KIND_PLE_KEY, KIND_PLE_VALUE, KIND_PLE_NORM_KEY,
                   KIND_PLE_NORM_QUERY, KIND_PLE_NORM_CONV, KIND_PLE_CONV,
                   KIND_PLE_MULTIPLIERS, KIND_PLE_HEAD_VOCABS,
                   KIND_PLE_HEAD_OFFSETS, KIND_PLE_NGRAM)

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
    # v2 hc/indexer/PLE-side tensors replicate: the stream vector is
    # replicated across ranks (each sublayer output is all-reduced before
    # the inject), so every rank runs the identical mixer math on the
    # identical inputs. The n-gram table is the exception (vocab-sharded).
    KIND_ATTN_HC_DOWN, KIND_ATTN_HC_UP, KIND_ATTN_HC_INJECT,
    KIND_MLP_HC_DOWN, KIND_MLP_HC_UP, KIND_MLP_HC_INJECT,
    KIND_INDEXER_QK, KIND_INDEXER_Q_NORM, KIND_INDEXER_K_NORM,
    KIND_MIXER_DOWN, KIND_MIXER_UP, KIND_MTP_MIXER_DOWN, KIND_MTP_MIXER_UP,
    KIND_PLE_KEY, KIND_PLE_VALUE, KIND_PLE_NORM_KEY, KIND_PLE_NORM_QUERY,
    KIND_PLE_NORM_CONV, KIND_PLE_CONV, KIND_PLE_MULTIPLIERS,
    KIND_PLE_HEAD_VOCABS, KIND_PLE_HEAD_OFFSETS,
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
    tensors = layer_count * 16 + gdn * 9 + full * 9
    if first_layer <= PLE_LAYER < first_layer + layer_count:
        tensors += 10
    if first_layer == 0:
        tensors += 1
    if first_layer + layer_count == LAYER_COUNT:
        tensors += 2 + 4 + 4 + 25 + (1 if first_layer != 0 else 0)
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
        if layer == PLE_LAYER:
            for kind in PLE_LAYER_KINDS:
                refs.append(TensorRef(kind, layer, layer_tensor_name(kind, layer)))
    if first_layer + layer_count == LAYER_COUNT:
        if first_layer != 0:
            refs.append(TensorRef(KIND_EMBEDDING, GLOBAL_LAYER, GLOBAL_TENSORS[KIND_EMBEDDING]))
        for kind in (KIND_FINAL_NORM, KIND_MIXER_DOWN, KIND_MIXER_UP,
                     KIND_LM_HEAD, KIND_MTP_FC, KIND_MTP_EMBED_NORM,
                     KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM,
                     KIND_MTP_MIXER_DOWN, KIND_MTP_MIXER_UP):
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

    def check_moe_split(self, ref: TensorRef, expert_start: int, expert_count: int) -> tuple[str, dict, int]:
        """fp8-official variant: validate the SPLIT per-expert tensors
        (experts.{e}.{gate,up,down}_proj.weight, F8_E4M3, plus their
        F32 weight_scale_inv planes) and return the anchor resolve for
        the slice's first expert. The pack layout stays fused
        expert-major; copy_fp8_official_experts does the gather."""
        split_name = {KIND_MOE_W1: "gate_proj", KIND_MOE_W3: "up_proj",
                      KIND_MOE_DOWN: "down_proj"}[ref.kind]
        rows_per_expert = ref.rows // expert_count
        fused = "gate_up_proj" if ref.kind in (KIND_MOE_W1, KIND_MOE_W3) else "down_proj"
        base = ref.name.replace("mlp.experts." + fused, "mlp.experts.{e}." + split_name + ".weight")
        anchor = None
        for e in range(expert_start, expert_start + expert_count):
            name = base.replace("{e}", str(e))
            shard, meta, offset = self.resolve(name)
            if meta["dtype"] != "F8_E4M3" or meta["shape"] != [rows_per_expert, ref.columns]:
                raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected F8_E4M3 [{rows_per_expert},{ref.columns}]")
            if anchor is None:
                anchor = (shard, meta, offset)
            sname = name[:-len(".weight")] + ".weight_scale_inv"
            s_meta = self.resolve(sname)[1]
            if s_meta["dtype"] not in ("F32", "BF16"):
                raise PackFailure(f"{sname}: scale dtype {s_meta['dtype']}, expected F32 or BF16")
            expect_scale = [rows_per_expert // 128, ref.columns // 128]
            if s_meta["shape"] != expect_scale:
                raise PackFailure(f"{sname}: scale shape {s_meta['shape']}, expected {expect_scale}")
        return anchor

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
        if ref.kind in (KIND_FINAL_NORM, KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM,
                        KIND_ATTENTION_NORM, KIND_MLP_NORM,
                        KIND_PLE_NORM_KEY, KIND_PLE_NORM_QUERY, KIND_PLE_NORM_CONV):
            shard, meta, offset = self.resolve(name)
            if meta["dtype"] != "BF16" or meta["shape"] != [4 * HIDDEN]:
                raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected BF16 [{4 * HIDDEN}] (4-stream group norm)")
            return shard, meta, offset
        if ref.kind in (KIND_ATTN_HC_DOWN, KIND_MLP_HC_DOWN, KIND_MIXER_DOWN, KIND_MTP_MIXER_DOWN):
            return super().check_shape(name, HC_LOWRANK, STREAM_WIDTH)
        if ref.kind in (KIND_ATTN_HC_UP, KIND_MLP_HC_UP, KIND_MIXER_UP, KIND_MTP_MIXER_UP):
            return super().check_shape(name, STREAM_WIDTH, HC_LOWRANK)
        if ref.kind in (KIND_ATTN_HC_INJECT, KIND_MLP_HC_INJECT):
            return super().check_shape(name, HC_STREAMS, STREAM_WIDTH)
        if ref.kind == KIND_INDEXER_QK:
            return super().check_shape(name, (INDEXER_HEADS + INDEXER_KV_HEADS) * INDEXER_HEAD_DIM, HIDDEN)
        if ref.kind in (KIND_INDEXER_Q_NORM, KIND_INDEXER_K_NORM):
            return super().check_shape(name, 1, INDEXER_HEAD_DIM)
        if ref.kind == KIND_PLE_KEY:
            return super().check_shape(name, STREAM_WIDTH, HIDDEN)
        if ref.kind == KIND_PLE_VALUE:
            return super().check_shape(name, HIDDEN, HIDDEN)
        if ref.kind == KIND_PLE_CONV:
            shard, meta, offset = self.resolve(name)
            if meta["dtype"] != "BF16" or meta["shape"] != [STREAM_WIDTH, 1, 4]:
                raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected BF16 [{STREAM_WIDTH},1,4]")
            return shard, meta, offset
        if ref.kind in (KIND_PLE_MULTIPLIERS, KIND_PLE_HEAD_VOCABS, KIND_PLE_HEAD_OFFSETS):
            shard, meta, offset = self.resolve(name)
            if meta["dtype"] != "I64" or meta["shape"] != [ref.columns]:
                raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected I64 [{ref.columns}]")
            return shard, meta, offset
        if ref.kind == KIND_PLE_NGRAM:
            shard, meta, offset = self.resolve(f"{LAYER_PREFIX}{PLE_LAYER}.ple.ple_embedding.ngram_embedding.shard_0.weight")
            if meta["dtype"] != "BF16" or meta["shape"] != [PLE_NGRAM_ROWS // 128, PLE_NGRAM_HEAD_DIM]:
                raise PackFailure(f"ngram shard_0: {meta['dtype']} {meta['shape']}, expected BF16 [{PLE_NGRAM_ROWS // 128},{PLE_NGRAM_HEAD_DIM}]")
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


class Fp8OfficialCheckShapeMixin:
    """Mixin for the fp8-official variant: MOE kinds validate the SPLIT
    per-expert tensors instead of the fused name; the PLE ngram table
    arrives as F8_E4M3 and widens losslessly to the BF16 wire format."""

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        if ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN):
            return self.check_moe_split(ref, 0, EXPERT_COUNT)
        if ref.kind == KIND_PLE_NGRAM:
            shard, meta, offset = self.resolve(
                f"{LAYER_PREFIX}{PLE_LAYER}.ple.ple_embedding.ngram_embedding.shard_0.weight")
            if meta["dtype"] == "F8_E4M3" and meta["shape"] == [PLE_NGRAM_ROWS // 128, PLE_NGRAM_HEAD_DIM]:
                ref.ple_ngram_f8 = True
                return shard, meta, offset
            if meta["dtype"] == "BF16" and meta["shape"] == [PLE_NGRAM_ROWS // 128, PLE_NGRAM_HEAD_DIM]:
                return shard, meta, offset
            raise PackFailure(f"ngram shard_0: {meta['dtype']} {meta['shape']}")
        return super().check_shape(ref)


class Fp8OfficialSource(Fp8OfficialCheckShapeMixin, SafetensorsSource):
    """The official fp8 release (qwen3.8-flash-next-fp8) source reader:
    split per-expert F8_E4M3 tensors + F32 weight_scale_inv planes,
    repackage-only passthrough into the fused pack layout."""


class Nvfp4OfficialCheckShapeMixin:
    """Mixin for the nvfp4-official variant: MOE kinds validate the SPLIT
    per-expert packed tensors (experts.{e}.{gate,up,down}_proj.weight, U8,
    plus F8_E4M3 weight_scale [rows, cols/16] and F32 input_scale planes)
    instead of the fused name."""

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        if ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN) and ref.layer != MTP_LAYER:
            proj = {KIND_MOE_W1: "gate_proj", KIND_MOE_W3: "up_proj",
                    KIND_MOE_DOWN: "down_proj"}[ref.kind]
            fused = "gate_up_proj" if ref.kind in (KIND_MOE_W1, KIND_MOE_W3) else "down_proj"
            base = ref.name.replace("mlp.experts." + fused, "mlp.experts.{e}." + proj)
            rows_per_expert = ref.rows // EXPERT_COUNT
            anchor = None
            packed_cols = ref.columns // 2
            scale_cols = ref.columns // 16
            for e in range(EXPERT_COUNT):
                name = base.replace("{e}", str(e)) + ".weight"
                shard, meta, offset = self.resolve(name)
                if meta["dtype"] != "U8" or meta["shape"] != [rows_per_expert, packed_cols]:
                    raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected U8 [{rows_per_expert},{packed_cols}]")
                if anchor is None:
                    anchor = (shard, meta, offset)
                sname = name[:-len(".weight")] + ".weight_scale"
                s_meta = self.resolve(sname)[1]
                if s_meta["dtype"] != "F8_E4M3" or s_meta["shape"] != [rows_per_expert, scale_cols]:
                    raise PackFailure(f"{sname}: {s_meta['dtype']} {s_meta['shape']}, expected F8_E4M3 [{rows_per_expert},{scale_cols}]")
            return anchor
        if ref.kind == KIND_PLE_NGRAM:
            # The ngram table arrives as F8_E4M3 in the nvfp4 release and
            # widens losslessly to the BF16 wire on copy (the same LUT
            # class as the fp8-official ngram path).
            return self.resolve(ref.name + ".shard_0.weight")
        return super().check_shape(ref)


class Nvfp4OfficialSource(Nvfp4OfficialCheckShapeMixin, SafetensorsSource):
    """The official nvfp4 release (qwen3.8-flash-next-nvfp4-radixark) source
    reader: split per-expert U8-packed e2m1 tensors + F8_E4M3 per-16 scale
    planes + F32 input scales, repackage-only passthrough into the fused
    pack layout under the NVFP4 wire code."""


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
    elif ref.kind == KIND_PLE_NGRAM:
        # Vocab-sharded n-gram table: rank r owns the contiguous row span
        # [r*rows/tp, (r+1)*rows/tp) of the head-major concatenated space,
        # which is exactly shards [r*128/tp, (r+1)*128/tp) of the 128
        # checkpoint shards (each shard is rows/128 rows).
        ref.rows = PLE_NGRAM_ROWS // tp_degree
        ref.ngram_shard_range = (tp_rank * (128 // tp_degree),
                                 (tp_rank + 1) * (128 // tp_degree))
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


def _e4m3_to_bf16_lut():
    """256-entry byte table: e4m3 code -> bf16 bits (lossless widening)."""
    import numpy as np
    raw = np.arange(256, dtype=np.uint8)
    sign = (raw & 0x80).astype(np.uint32) << 24
    exp = ((raw >> 3) & 0xF).astype(np.int32)
    man = (raw & 0x7).astype(np.uint32)
    # e4m3 bias 7 -> f32 exponent field; subnormals flow through f32 math.
    value = np.where(exp == 0,
                     (man / 8.0) * 2.0 ** -6,
                     (1.0 + man / 8.0) * 2.0 ** (exp - 7))
    bits = (np.frombuffer(value.astype("<f4").tobytes(), dtype=np.uint32)
            | sign).astype(np.uint32) >> 16
    return np.where((raw & 0x7F) == 0x7F, np.uint32(0x7FC0), bits.astype(np.uint32)).astype("<u2")

_E4M3_BF16_LUT = None

def copy_ple_ngram_f8_widen(source, ref: TensorRef, offset: int, out) -> None:
    """PLE ngram stored as F8_E4M3 in the official fp8 source: stream the
    raw bytes and widen each value losslessly to BF16 (the pack's wire
    format for this tensor). Chunked + page-cache-evicting per the
    memory law."""
    global _E4M3_BF16_LUT
    import numpy as np
    if _E4M3_BF16_LUT is None:
        _E4M3_BF16_LUT = _e4m3_to_bf16_lut()
    shard, _, off = source.resolve(ref.name)
    total = ref.rows * ref.columns
    with (source.root / shard).open("rb") as f:
        fd = f.fileno()
        remaining = total
        while remaining > 0:
            step = min(remaining, 512 * 1024)
            raw = os.pread(fd, step, off)
            if len(raw) != step:
                raise PackFailure(f"short read on {ref.name}")
            codes = np.frombuffer(raw, dtype=np.uint8)
            out.write(_E4M3_BF16_LUT[codes].tobytes())
            try:
                os.posix_fadvise(fd, off, step, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            off += step
            remaining -= step

def copy_sharded_bf16(source: SafetensorsSource, ref: TensorRef, offset: int, out) -> None:
    import numpy as np
    # I64 hash constants: raw little-endian copy, never converted.
    if ref.weight_format == WEIGHT_I64:
        with (source.root / source.weight_map[ref.name]).open("rb") as file:
            file.seek(source.resolve(ref.name)[2])
            raw = file.read(ref.columns * 8)
        if len(raw) != ref.columns * 8:
            raise PackFailure(f"short read on {ref.name}")
        out.write(raw)
        return
    # N-gram table: stream this rank's contiguous shard span in row order.
    if ref.kind == KIND_PLE_NGRAM:
        shard_start, shard_end = getattr(ref, "ngram_shard_range", (0, 128))
        for shard_index in range(shard_start, shard_end):
            shard_name = f"{LAYER_PREFIX}{PLE_LAYER}.ple.ple_embedding.ngram_embedding.shard_{shard_index}.weight"
            with (source.root / source.weight_map[shard_name]).open("rb") as file:
                file.seek(source.resolve(shard_name)[2])
                remaining = (PLE_NGRAM_ROWS // 128) * PLE_NGRAM_HEAD_DIM * BF16_BYTES
                while remaining > 0:
                    chunk = file.read(min(remaining, CHUNK_BYTES))
                    if len(chunk) != min(remaining, CHUNK_BYTES):
                        raise PackFailure(f"short read on {shard_name}")
                    remaining -= len(chunk)
                    out.write(chunk)
        return
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
        # whole-tensor stream (handles f32 widening and conv squeeze),
        # small chunks + fadvise per the node-memory law.
        elements = ref.rows * ref.columns
        source_bytes = elements * BF16_BYTES
        with path.open("rb") as file:
            fd = file.fileno()
            offset_left = source_bytes
            position = offset
            while offset_left > 0:
                step = min(offset_left, 512 * 1024)
                chunk = os.pread(fd, step, position)
                if len(chunk) != step:
                    raise PackFailure(f"short read on {ref.name}")
                offset_left -= step
                position += step
                if ref.weight_format == WEIGHT_BF16:
                    out.write(chunk)
                else:
                    widened = bytearray(step * 2)
                    widened[2::4] = chunk[0::2]
                    widened[3::4] = chunk[1::2]
                    out.write(widened)
                try:
                    os.posix_fadvise(fd, position - step, step, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
        return
    if ref.kind == KIND_PLE_NGRAM:
        # Stream this rank's contiguous shard span (128/tp shards) in row
        # order - a raw byte stream, no conversion.
        shard_start, shard_end = getattr(ref, "ngram_shard_range", (0, 128))
        for shard_index in range(shard_start, shard_end):
            shard_name = f"{LAYER_PREFIX}{PLE_LAYER}.ple.ple_embedding.ngram_embedding.shard_{shard_index}.weight"
            path = source.root / source.weight_map[shard_name]
            with path.open("rb") as file:
                remaining = (PLE_NGRAM_ROWS // 128) * PLE_NGRAM_HEAD_DIM * BF16_BYTES
                file.seek(source.resolve(shard_name)[2])
                while remaining > 0:
                    chunk = file.read(min(remaining, CHUNK_BYTES))
                    if len(chunk) != min(remaining, CHUNK_BYTES):
                        raise PackFailure(f"short read on {shard_name}")
                    remaining -= len(chunk)
                    out.write(chunk)
        return
    if ref.weight_format == WEIGHT_I64:
        path = source.root / source.weight_map[ref.name]
        with path.open("rb") as file:
            file.seek(source.resolve(ref.name)[2])
            raw = file.read(ref.columns * 8)
        if len(raw) != ref.columns * 8:
            raise PackFailure(f"short read on {ref.name}")
        out.write(raw)
        return
    full = read_source_matrix(source, ref.name)
    if full.ndim == 3 and ref.kind == KIND_PLE_CONV:
        full = full[:, 0, :]
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


def pump_read(fd, offset: int, length: int, out) -> None:
    """Stream fd[offset:offset+length) to out in small chunks, evicting
    each chunk from the page cache (memory law: no big warm streams)."""
    remaining = length
    while remaining > 0:
        step = min(remaining, 512 * 1024)
        raw = os.pread(fd, step, offset)
        if len(raw) != step:
            raise PackFailure(f"short read at {offset}")
        out.write(raw)
        try:
            os.posix_fadvise(fd, offset, step, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        offset += step
        remaining -= step

def copy_nvfp4_official_experts(source, ref: TensorRef, out) -> None:
    """nvfp4-official arm: gather the rank's SPLIT per-expert U8-packed
    e2m1 tensors into the fused expert-major pack layout, verbatim bytes
    + verbatim F8_E4M3 per-16 scale planes. Two-pass (payloads, then
    scales) so nothing accumulates in RAM. Repackage-only."""
    expert_start, expert_count = getattr(ref, "expert_slice", (0, EXPERT_COUNT))
    rows_per_expert = ref.rows // expert_count
    split_name = {KIND_MOE_W1: "gate_proj", KIND_MOE_W3: "up_proj",
                  KIND_MOE_DOWN: "down_proj"}[ref.kind]
    fused = "gate_up_proj" if ref.kind in (KIND_MOE_W1, KIND_MOE_W3) else "down_proj"
    split_suffix = split_name + ".weight"
    base = ref.name.replace("mlp.experts." + fused, "mlp.experts.{e}." + split_suffix)
    scale_suffix = split_name + ".weight_scale"
    scale_base = ref.name.replace("mlp.experts." + fused, "mlp.experts.{e}." + scale_suffix)
    payload_fds, scale_fds = [], []
    try:
        for e in range(expert_start, expert_start + expert_count):
            shard, meta, off = source.resolve(base.replace("{e}", str(e)))
            f = (source.root / shard).open("rb")
            payload_fds.append((f, off))
        for f, off in payload_fds:
            pump_read(f.fileno(), off, rows_per_expert * ref.columns, out)
    finally:
        for f, _ in payload_fds:
            f.close()
    s_rows = rows_per_expert
    s_cols = ref.columns // 16
    try:
        for e in range(expert_start, expert_start + expert_count):
            s_shard, _, s_off = source.resolve(
                base.replace("{e}", str(e))[:-len(".weight")] + ".weight_scale")
            sf = (source.root / s_shard).open("rb")
            scale_fds.append((sf, s_off))
        for sf, s_off in scale_fds:
            pump_read(sf.fileno(), s_off, s_rows * s_cols, out)
    finally:
        for sf, _ in scale_fds:
            sf.close()

def copy_fp8_official_experts(source, ref: TensorRef, offset: int, out) -> None:
    """fp8-official arm: gather the rank's SPLIT per-expert F8_E4M3
    tensors into the fused expert-major pack layout, verbatim bytes +
    verbatim F32 weight_scale_inv planes. Two-pass (payloads, then
    scales) so nothing accumulates in RAM. Repackage-only."""
    expert_start, expert_count = getattr(ref, "expert_slice", (0, EXPERT_COUNT))
    rows_per_expert = ref.rows // expert_count
    split_name = {KIND_MOE_W1: "gate_proj", KIND_MOE_W3: "up_proj",
                  KIND_MOE_DOWN: "down_proj"}[ref.kind]
    fused = "gate_up_proj" if ref.kind in (KIND_MOE_W1, KIND_MOE_W3) else "down_proj"
    split_suffix = split_name + ".weight"
    base = ref.name.replace("mlp.experts." + fused, "mlp.experts.{e}." + split_suffix)
    scale_suffix = split_name + ".weight_scale_inv"
    scale_base = ref.name.replace("mlp.experts." + fused, "mlp.experts.{e}." + scale_suffix)
    payload_fds, scale_fds = [], []
    try:
        for e in range(expert_start, expert_start + expert_count):
            shard, meta, off = source.resolve(base.replace("{e}", str(e)))
            f = (source.root / shard).open("rb")
            payload_fds.append((f, off))
        for f, off in payload_fds:
            pump_read(f.fileno(), off, rows_per_expert * ref.columns, out)
    finally:
        for f, _ in payload_fds:
            f.close()
    # Scales: source planes are F32 or BF16; the pack's F32B128 wire plane
    # is F32, so BF16 values widen losslessly on the way through.
    s_rows = rows_per_expert // FP8_BLOCK
    s_cols = ref.columns // FP8_BLOCK
    try:
        for e in range(expert_start, expert_start + expert_count):
            s_shard, s_meta, s_off = source.resolve(scale_base.replace("{e}", str(e)))
            sf = (source.root / s_shard).open("rb")
            scale_fds.append((sf, s_off, s_meta["dtype"]))
        for sf, s_off, s_dtype in scale_fds:
            remaining = s_rows * s_cols * F32_BYTES
            offset = s_off
            while remaining > 0:
                step = min(remaining, 512 * 1024)
                if s_dtype == "BF16":
                    half = step // 2
                    raw = sf.read(half)
                    if len(raw) != half:
                        raise PackFailure("short read on weight_scale_inv")
                    widened = bytearray(step)
                    widened[2::4] = raw[0::2]
                    widened[3::4] = raw[1::2]
                    out.write(bytes(widened))
                else:
                    raw = sf.read(step)
                    if len(raw) != step:
                        raise PackFailure("short read on weight_scale_inv")
                    out.write(raw)
                offset += step
                remaining -= step
    finally:
        for sf, _, _ in scale_fds:
            sf.close()

def copy_ngram_f8_widen(source, ref: TensorRef, tp_degree: int, tp_rank: int, out) -> None:
    """PLE ngram table under the fp8-official source: the table spans 128
    F8 shards (PLE_NGRAM_ROWS//128 rows each); this rank's TP slice is
    shards_per_rank consecutive shards, widened losslessly F8->BF16 via
    the LUT onto the pack's BF16 wire. Chunked + fadvise per the memory
    law. Repackage-only."""
    global _E4M3_BF16_LUT
    import numpy as np
    if _E4M3_BF16_LUT is None:
        _E4M3_BF16_LUT = _e4m3_to_bf16_lut()
    shards_per_rank = 128 // tp_degree
    first = tp_rank * shards_per_rank
    rows_per_shard = PLE_NGRAM_ROWS // 128
    raw_per_shard = rows_per_shard * PLE_NGRAM_HEAD_DIM
    for i in range(first, first + shards_per_rank):
        shard, meta, off = source.resolve(ref.name + f".shard_{i}.weight")
        with (source.root / shard).open("rb") as f:
            fd = f.fileno()
            remaining = raw_per_shard
            pos = off
            while remaining > 0:
                step = min(remaining, 512 * 1024)
                raw = os.pread(fd, step, pos)
                if len(raw) != step:
                    raise PackFailure(f"short read on {ref.name}.shard_{i}")
                codes = np.frombuffer(raw, dtype=np.uint8)
                out.write(_E4M3_BF16_LUT[codes].tobytes())
                try:
                    os.posix_fadvise(fd, pos, step, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
                pos += step
                remaining -= step

def quantize_experts(source: SafetensorsSource, ref: TensorRef, expert_format: str, out) -> None:
    """Read the fused per-layer expert tensors [E, 2I, H] / [E, H, I],
    split w1/w3 per expert, quantize per 128x128 block, and stack the
    rank's experts expert-major with the scale plane after the payload.
    --expert-format bf16 writes the source BF16 bits verbatim (the
    quantization policy: repackage-only, never quantize)."""
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
    if expert_format == "bf16":
        out.write(np.ascontiguousarray(matrix, dtype="<u2").tobytes())
        return
    payload, scales = quantize_fp8_blocks(bf16_to_f32_matrix(matrix), expert_format)
    out.write(payload)
    out.write(scales)


def convert(checkpoint: Path, output: Path, first_layer: int, layer_count: int,
            receipt: dict, dry_run: bool, tp_degree: int, tp_rank: int,
            expert_format: str) -> dict:
    import numpy as np  # noqa: F401  (quantization paths import lazily)
    source_cls = {"fp8-official": Fp8OfficialSource,
                  "nvfp4-official": Nvfp4OfficialSource}.get(expert_format, SafetensorsSource)
    source = source_cls(checkpoint)
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
    # The routed experts' WIRE format (natural is F32B128; the CLI flag
    # swaps in the per-row MX plane, the policy-mandated BF16 repackage,
    # or the verbatim nvfp4-packed bytes). The plan below must price
    # payload and scale bytes by the WIRE format - the writer emits
    # exactly this layout.
    expert_wire_format = {"fp8-f32b128": WEIGHT_FP8_F32B128,
                          "fp8-e8m0b128": WEIGHT_FP8_E8M0B128,
                          "fp8-official": WEIGHT_FP8_F32B128,
                          "nvfp4-official": 8,
                          "bf16": WEIGHT_BF16}[expert_format]
    plans = []
    cursor = 0
    for ref in refs:
        wire = expert_wire_format if ref.weight_format == WEIGHT_FP8_F32B128 else ref.weight_format
        if wire == 8:
            # NVFP4 wire: U8-packed e2m1 payloads (2 values/byte) +
            # e4m3 scale bytes per 16 values + the per-tensor F32
            # input scale. Verbatim passthrough of the release bytes.
            payload_bytes = ref.rows * (ref.columns // 2)
            scale_bytes = ref.rows * (ref.columns // 16) + 4
        elif wire in (WEIGHT_FP8_F32B128, WEIGHT_FP8_E8M0B128):
            payload_bytes = ref.rows * ref.columns
            # F32B128: one f32 per 128x128 tile; E8M0B128: one exponent byte
            # per (row, 128-column block) - the per-row MX plane the module's
            # grouped expert kernels decode.
            scale_bytes = ((ref.rows // FP8_BLOCK) * (ref.columns // FP8_BLOCK) * F32_BYTES
                           if wire == WEIGHT_FP8_F32B128
                           else ref.rows * (ref.columns // FP8_BLOCK))
        elif ref.weight_format == WEIGHT_I64:
            payload_bytes = ref.rows * ref.columns * 8
            scale_bytes = 0
        else:
            payload_bytes = ref.rows * ref.columns * (BF16_BYTES if wire == WEIGHT_BF16 else F32_BYTES)
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
            FP8_BLOCK if (ref.weight_format == WEIGHT_FP8_F32B128 and expert_wire_format != WEIGHT_BF16) or ref.weight_format == WEIGHT_FP8_E8M0B128 else 0,
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
            elif expert_format == "fp8-official" and ref.kind == KIND_PLE_NGRAM:
                copy_ngram_f8_widen(source, ref, tp_degree, tp_rank, temp)
            elif expert_format == "nvfp4-official" and ref.kind in (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN) and ref.layer != MTP_LAYER:
                copy_nvfp4_official_experts(source, ref, temp)
            elif ref.weight_format in (WEIGHT_FP8_F32B128, WEIGHT_FP8_E8M0B128):
                if expert_format == "fp8-official":
                    copy_fp8_official_experts(source, ref, temp)
                else:
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
    parser.add_argument("--expert-format", choices=("fp8-f32b128", "fp8-e8m0b128", "bf16", "fp8-official", "nvfp4-official"), default="fp8-f32b128",
                        help="fp8-official = the official fp8 release's split experts pass through verbatim (repackage-only); nvfp4-official = the official nvfp4 release's split U8-packed experts + e4m3 scale planes verbatim")
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
        "hc_semantics": [
            "v2 packs the EXACT reference semantics (modeling_qwen4_exp.py, sha256 77fec77d): full [4H] hc_norm on the norm slots, per-sublayer low-rank mixers + block_inject, the hyper_connection_mixer readout (use_combine=False), the attention indexer, and the layer-1 PLE block",
            "PLE ngram table is vocab-sharded bf16 (operator decision 2026-08-28: 23.84 GiB/rank at TP4, no quantization loss)",
            "MTP: pre_fc_norm_hidden [4H] group-norms the streams, the mtp.hyper_connection_mixer mean-mixes (the publisher reference ignores mtp.* keys, so this composition is the in-house EAGLE convention)",
        ],
        "unmapped_checkpoint_tensors": {
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
