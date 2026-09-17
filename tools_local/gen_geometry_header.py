#!/usr/bin/env python3
"""Geometry-header generator: the recipe compiler v0 (W4 redundancy lane).

model_contracts/<family>_authoritative.json -> the family geometry header
(model-families/<family>/include/sparkpipe/spark_<family>_model.h) and the
family's serving-adapter descriptor constants blob. Every geometry number in
the emitted header is read from the contract (strict indexing - a missing key
is an error, never a default); prose and derived-macro structure are owned by
this file's family templates, the same split the dsv4/k3 contract generators
use. Provenance for constants that are module/deployment policy rather than
model geometry is recorded in FAMILY_POLICY below.

Proof discipline (docs/HOUSECLEANING_PLAN.md W4.4): where a hand-written
original exists, --check must reproduce it byte-identical before cutover.
qwen38_27b is the byte-identity proof; glm5_next and qwen4_flash were cut
over after diff review (see docs/AGENT_LANE_BRIEFS/reports/).

Usage:
    python3 tools/gen_geometry_header.py --family qwen38_27b [--check]
    python3 tools/gen_geometry_header.py --family glm5_next  [--check]
    python3 tools/gen_geometry_header.py --family qwen4_flash [--check]
    python3 tools/gen_geometry_header.py --family qwen38_27b --emit-adapter-constants [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "qwen38_27b": {
        "contract": "model_contracts/qwen38_27b_authoritative.json",
        "header": "model-families/qwen38_27b/include/sparkpipe/spark_qwen38_27b_model.h",
    },
    "glm5_next": {
        "contract": "model_contracts/glm53_flash_authoritative.json",
        "header": "model-families/glm5_next/include/sparkpipe/spark_glm5_next_model.h",
    },
    "qwen4_flash": {
        "contract": "model_contracts/qwen4_flash_authoritative.json",
        "header": "model-families/qwen4_flash/include/sparkpipe/spark_qwen4_flash_model.h",
    },
}

ADAPTER_CONSTANTS = {
    "qwen38_27b": "model-families/qwen38_27b/include/sparkpipe/spark_qwen38_27b_serving_constants.h",
}

# Constants that are module/deployment policy, not checkpoint geometry; the
# contract does not carry them. Each entry names its owner so the split stays
# auditable. Values land in the emitted header verbatim.
FAMILY_POLICY = {
    "qwen38_27b": {
        "gdn_chunk_tokens": 64,       # module chunk width (kernel walk granularity)
    },
    "glm5_next": {
        "kv_pool_tokens": 4194304,                # deployment KV policy
        "restricted_vocab_count": 256,            # deployment sampling policy
        "max_prefill_tokens_per_dispatch": 256,   # deployment batching policy
        "kda_qk_l2norm": 1,                       # kernel semantics (k3 donor)
        "kda_full_rank_gate": 1,                  # dt_bias [8192] census
        "kda_state_element_bytes": 4,             # fp32 recurrent state
        "kda_short_conv_bf16_bytes": 2,
        "mla_use_nope": 1,                        # uses_nope_only
        "index_norm_epsilon": 1e-06,              # indexer LayerNorm (reference default)
        "hc_scale_count": 3,                      # hc_attn_scale [3] census
        "moe_w1_component_count": 2,              # fused gate|up
        "kv_bits": 16,                            # glm52-lineage page policy
        "kv_page_slots": 64,
        "bf16_bytes": 2,
    },
    "qwen4_flash": {
        "mxfp4_group_size": 32,      # format-4/6 MX plane geometry (pack codec)
        "fp8_block": 128,            # fp8 block-128 scale plane
        "swiglu_limit": 10.0,        # activation clamp (family activation silu)
        "gdn_chunk_tokens": 64,      # module chunk width
        "bf16_bytes": 2,
    },
}


def load_contract(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def u(value: int) -> str:
    return f"{value}u"


def f32(value: float) -> str:
    text = repr(float(value))
    return f"{text}f"


def scientific(value: float) -> str:
    return f"{value:.0e}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def render_qwen38_27b(c: dict) -> str:
    model, hybrid, gdn, attn = c["model"], c["hybrid_attention"], c["gdn"], c["attention"]
    cache = c["cache"]
    policy = FAMILY_POLICY["qwen38_27b"]
    require(hybrid["gdn_layer_count"] + hybrid["full_layer_count"] == model["layer_count"],
            "qwen38_27b hybrid layer partition must cover the stack")
    require(model["rms_norm_epsilon"] == 1e-06, "qwen38_27b epsilon literal formatting below assumes 1e-06")

    hidden, layers, vocab = u(model["hidden_dimension"]), u(model["layer_count"]), u(model["vocabulary_size"])
    heads, kv_heads, head_dim = u(attn["query_head_count"]), u(attn["kv_head_count"]), u(attn["head_dimension"])
    gdn_key_heads, gdn_value_heads = u(gdn["key_head_count"]), u(gdn["value_head_count"])
    max_ctx = u(model["maximum_context_tokens"])
    eps = f"{model['rms_norm_epsilon']:.0e}"
    ffn = u(model["dense_intermediate_dimension"])
    mtp = u(model["mtp_layer_count"])
    period, phase = u(hybrid["period"]), u(hybrid["full_phase"])
    gdn_layers, full_layers = u(hybrid["gdn_layer_count"]), u(hybrid["full_layer_count"])
    gdn_key_dim, gdn_value_dim = u(gdn["key_dimension"]), u(gdn["value_dimension"])
    conv_kernel = u(gdn["short_conv_kernel"])
    rope_dim = u(attn["rope_dimension"])
    theta = repr(float(attn["rope_theta"]))
    chunk = u(policy["gdn_chunk_tokens"])
    page = u(cache["kv_page_slots"])

    return f"""// Qwen 3.6's model geometry for the host tiers, mirroring the values in
// inference/llms/qwen_3_6/config.h. The firmware config is the source
// of truth; a conformance check holds the two in lockstep.
#ifndef SPARKPIPE_SPARK_QWEN38_27B_MODEL_H
#define SPARKPIPE_SPARK_QWEN38_27B_MODEL_H

#define SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION {hidden}
#define SPARK_QWEN38_27B_MODEL_LAYER_COUNT {layers}
#define SPARK_QWEN38_27B_MODEL_VOCAB_COUNT {vocab}
#define SPARK_QWEN38_27B_MODEL_ATTENTION_HEAD_COUNT {heads}
#define SPARK_QWEN38_27B_MODEL_KV_HEAD_COUNT {kv_heads}
#define SPARK_QWEN38_27B_MODEL_HEAD_DIMENSION {head_dim}
#define SPARK_QWEN38_27B_MODEL_GDN_KEY_HEAD_COUNT {gdn_key_heads}
#define SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT {gdn_value_heads}

// The resident decode stage module predates the shim above and speaks the
// wider geometry vocabulary below. Every value is either a config.h constant
// under its second name or derived from them; the two additions with no
// config.h counterpart are MAXIMUM_CONTEXT_TOKENS (checkpoint config
// max_position_embeddings) and GDN_CHUNK_TOKENS (the module's chunk width).
#define SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT SPARK_QWEN38_27B_MODEL_VOCAB_COUNT
#define SPARK_QWEN38_27B_MODEL_MAXIMUM_CONTEXT_TOKENS {max_ctx}
#define SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON {eps}f
#define SPARK_QWEN38_27B_MODEL_FFN_INTERMEDIATE_DIMENSION {ffn}
#define SPARK_QWEN38_27B_MODEL_MTP_LAYER_COUNT {mtp}

#define SPARK_QWEN38_27B_MODEL_ATTENTION_PERIOD {period}
#define SPARK_QWEN38_27B_MODEL_FULL_ATTENTION_PHASE {phase}
#define SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(layer_index) \\
	(((layer_index) % SPARK_QWEN38_27B_MODEL_ATTENTION_PERIOD) != SPARK_QWEN38_27B_MODEL_FULL_ATTENTION_PHASE)
#define SPARK_QWEN38_27B_MODEL_GDN_LAYER_COUNT {gdn_layers}
#define SPARK_QWEN38_27B_MODEL_FULL_ATTENTION_LAYER_COUNT {full_layers}

#define SPARK_QWEN38_27B_MODEL_GDN_HEAD_KEY_DIMENSION {gdn_key_dim}
#define SPARK_QWEN38_27B_MODEL_GDN_HEAD_VALUE_DIMENSION {gdn_value_dim}
#define SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD \\
	(SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT / SPARK_QWEN38_27B_MODEL_GDN_KEY_HEAD_COUNT)
#define SPARK_QWEN38_27B_MODEL_GDN_CONV_KERNEL {conv_kernel}
#define SPARK_QWEN38_27B_MODEL_GDN_QK_DIMENSION \\
	(SPARK_QWEN38_27B_MODEL_GDN_KEY_HEAD_COUNT * SPARK_QWEN38_27B_MODEL_GDN_HEAD_KEY_DIMENSION)
#define SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION \\
	(SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN38_27B_MODEL_GDN_HEAD_VALUE_DIMENSION)
#define SPARK_QWEN38_27B_MODEL_GDN_CONV_CHANNELS \\
	((2u * SPARK_QWEN38_27B_MODEL_GDN_QK_DIMENSION) + SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION)
#define SPARK_QWEN38_27B_MODEL_GDN_CONV_TAIL_COLUMNS (SPARK_QWEN38_27B_MODEL_GDN_CONV_KERNEL - 1u)
#define SPARK_QWEN38_27B_MODEL_GDN_CHUNK_TOKENS {chunk}

#define SPARK_QWEN38_27B_MODEL_ATTN_QUERY_HEAD_COUNT SPARK_QWEN38_27B_MODEL_ATTENTION_HEAD_COUNT
#define SPARK_QWEN38_27B_MODEL_ATTN_KV_HEAD_COUNT SPARK_QWEN38_27B_MODEL_KV_HEAD_COUNT
#define SPARK_QWEN38_27B_MODEL_ATTN_HEAD_DIMENSION SPARK_QWEN38_27B_MODEL_HEAD_DIMENSION
#define SPARK_QWEN38_27B_MODEL_ATTN_ROPE_DIMENSION {rope_dim}
#define SPARK_QWEN38_27B_MODEL_ATTN_ROPE_THETA {theta}f
#define SPARK_QWEN38_27B_MODEL_ATTN_QUERY_DIMENSION \\
	(SPARK_QWEN38_27B_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_QWEN38_27B_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_QWEN38_27B_MODEL_ATTN_KV_DIMENSION \\
	(SPARK_QWEN38_27B_MODEL_ATTN_KV_HEAD_COUNT * SPARK_QWEN38_27B_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_QWEN38_27B_MODEL_ATTN_CACHE_TOKEN_ELEMENTS \\
	(2u * SPARK_QWEN38_27B_MODEL_ATTN_KV_DIMENSION)

#define SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES 2u
#define SPARK_QWEN38_27B_MODEL_HIDDEN_BF16_BYTES \\
	(SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES)

#endif
"""


def render_glm5_next(c: dict) -> str:
    model, hybrid, kda, mla = c["model"], c["hybrid_attention"], c["kda"], c["mla"]
    index, hc, moe, tokens = c["indexer"], c["hyper_connections"], c["moe"], c["tokens"]
    precision = c["precision"]
    mtp = c["mtp"]
    policy = FAMILY_POLICY["glm5_next"]
    require(hybrid["kda_layer_count"] + hybrid["dsa_layer_count"] == model["layer_count"],
            "glm5_next hybrid layer partition must cover the stack")
    require(mla["qk_rope_head_dimension"] == 0, "glm5_next is the rope-0 MLA instantiation")
    require(kda["checkpoint_tensor_shapes"]["f_a_proj.weight"][1][0] == 128,
            "glm5_next low-rank gate bottleneck moved in the checkpoint")

    hidden, layers = u(model["hidden_dimension"]), u(model["layer_count"])
    mtp_index = u(mtp["layer_index"])
    vocab = u(model["vocabulary_size"])
    max_ctx = u(model["maximum_context_tokens"])
    eps = f"{model['rms_norm_epsilon']:.0e}"
    swiglu = repr(float(model["swiglu_limit"]))
    eos, user = u(tokens["end_of_text"]), u(tokens["user"])
    obs, pad = u(tokens["observation"]), u(tokens["pad"])
    period, phase = u(hybrid["period"]), u(hybrid["global_phase"])
    kda_layers, dsa_layers = u(hybrid["kda_layer_count"]), u(hybrid["dsa_layer_count"])
    kda_heads = u(kda["head_count"])
    kda_dim = u(kda["head_dimension"])
    conv = u(kda["short_conv_kernel"])
    gate_lb = repr(float(kda["gate_lower_bound"]))
    bottleneck = u(kda["checkpoint_tensor_shapes"]["f_a_proj.weight"][1][0])
    mla_heads = u(mla["query_head_count"])
    q_a = u(mla["query_lora_rank"])
    latent = u(mla["kv_lora_rank"])
    nope = u(mla["qk_nope_head_dimension"])
    rope = u(mla["qk_rope_head_dimension"])
    value_dim = u(mla["value_head_dimension"])
    qk_scale = repr(mla["value_head_dimension"] ** -0.5)
    idx_heads = u(index["head_count"])
    idx_dim = u(index["head_dimension"])
    top_k = u(index["top_k"])
    kpool = u(index["kpool"])
    idx_softmax = repr(index["head_dimension"] ** -0.5)
    idx_head_scale = repr(index["head_count"] ** -0.5)
    idx_eps = f"{policy['index_norm_epsilon']:.0e}"
    hc_mult = u(hc["hc_mult"])
    hc_iter = u(hc["sinkhorn_iterations"])
    hc_eps = f"{hc['epsilon']:.0e}"
    hc_scales = u(policy["hc_scale_count"])
    experts = u(moe["routed_expert_count"])
    topk = u(moe["experts_per_token"])
    shared = u(moe["shared_expert_count"])
    inter = u(moe["expert_intermediate_dimension"])
    dense_inter = u(moe["dense_intermediate_dimension"])
    first_dense = u(moe["first_dense_layer_count"])
    scaling = repr(float(moe["routed_scaling_factor"]))
    fp8_block = u(precision["weight_block_size"][0])
    kv_bits = u(policy["kv_bits"])
    page = u(policy["kv_page_slots"])
    norm_topk = u(1 if moe["normalize_selected_probabilities"] else 0)
    use_nope = u(1 if mla["uses_nope_only"] else 0)
    w1c = u(policy["moe_w1_component_count"])
    l2norm = u(policy["kda_qk_l2norm"])
    full_rank = u(policy["kda_full_rank_gate"])
    state_bytes = u(policy["kda_state_element_bytes"])
    bf16 = u(policy["bf16_bytes"])
    pool = u(policy["kv_pool_tokens"])
    restricted = u(policy["restricted_vocab_count"])
    prefill = u(policy["max_prefill_tokens_per_dispatch"])

    return f"""/* Generated by tools/gen_geometry_header.py from
 * model_contracts/glm53_flash_authoritative.json - do not edit by hand.
 * Cutover 2026-08-27 (W4 redundancy lane): diff-reviewed against the
 * hand-written original; the one content change is the removal of a
 * duplicated KDA_LOW_RANK_GATE_BOTTLENECK define. */
// GLM 5.3 Flash (family glm5_next) geometry for the host tiers.
//
// One geometry source for the family: every literal below is held against
// model_contracts/glm53_flash_authoritative.json by
// tests/test_glm5_next_geometry.py in lockstep, and every shape claim was
// censused from the checkpoint's safetensors headers (76,108 tensors; the
// evidence lives in the contract's checkpoint_tensor_shapes sections).
//
// The family is an ASSEMBLY of three donors:
//   glm52 - MLA projections + DSA indexer + MoE/router + FP8 [128,128] spine
//   k3    - KDA linear attention (kimi delta rule, chunk 64 + recurrent)
//   dsv4  - hyper-connections (mHC) + the kpool compressor mechanism
// The three real deltas are called out where they occur: rope-0 MLA, the
// checkpoint->pack name mapping (model-families/glm5_next/name_map.json),
// and the hybrid 34 KDA / 11 DSA dispatch.
//
// Reference semantics: transformers models/glm5_next/modeling_glm5_next.py
// (text stack only; the vision tower is out of scope for this lane).
#ifndef SPARKPIPE_SPARK_GLM5_NEXT_MODEL_H
#define SPARKPIPE_SPARK_GLM5_NEXT_MODEL_H

#include <stdint.h>

/* -- model ------------------------------------------------------------------ */
#define SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION {hidden}
#define SPARK_GLM5_NEXT_MODEL_LAYER_COUNT {layers}          /* weight layers 0..44; MTP is 45 */
#define SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX {mtp_index}
#define SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT {vocab}
#define SPARK_GLM5_NEXT_MODEL_MAXIMUM_CONTEXT_TOKENS {max_ctx}
#define SPARK_GLM5_NEXT_MODEL_RMS_NORM_EPSILON {eps}f
#define SPARK_GLM5_NEXT_MODEL_SWIGLU_LIMIT {swiglu}f
#define SPARK_GLM5_NEXT_MODEL_END_OF_TEXT_TOKEN_ID {eos}
#define SPARK_GLM5_NEXT_MODEL_USER_TOKEN_ID {user}
#define SPARK_GLM5_NEXT_MODEL_OBSERVATION_TOKEN_ID {obs}
#define SPARK_GLM5_NEXT_MODEL_PAD_TOKEN_ID {pad}

/* Deployment policy (not model constants). */
#define SPARK_GLM5_NEXT_MODEL_KV_POOL_TOKENS {pool}
#define SPARK_GLM5_NEXT_MODEL_RESTRICTED_VOCAB_COUNT {restricted}
#define SPARK_GLM5_NEXT_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH {prefill}

/* -- hybrid dispatch (DELTA 3) -----------------------------------------------
 * 45 weight layers: 34 KDA linear attention + 11 DSA at layers 3, 7, ..., 43
 * (a 3-KDA head, then every 4th). The checkpoint's kda_layers /
 * full_attn_layers lists agree with this closed form; the layer-45 MTP
 * layer is a DSA layer WITHOUT hyper-connections and is dispatched by the
 * speculative path, not this macro. */
#define SPARK_GLM5_NEXT_MODEL_ATTENTION_PERIOD {period}
#define SPARK_GLM5_NEXT_MODEL_GLOBAL_ATTENTION_PHASE {phase}
#define SPARK_GLM5_NEXT_MODEL_KDA_LAYER_COUNT {kda_layers}
#define SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT {dsa_layers}
#define SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer_index) \\
	(((layer_index) % SPARK_GLM5_NEXT_MODEL_ATTENTION_PERIOD) != \\
	 SPARK_GLM5_NEXT_MODEL_GLOBAL_ATTENTION_PHASE)

/* -- KDA linear attention (k3 donor) ----------------------------------------
 * kimi delta rule: 64 heads x 128 key/value, qk L2-normalized in kernel,
 * beta per head through sigmoid, decay per head-PER-CHANNEL (full rank),
 * state fp32 heads x key x value. DELTA vs the released K3 checkpoint:
 * the output gate is the LOW-RANK two-stage g_a[128]->g_b[8192] form with a
 * "safe" sigmoid forget gate; K3 shipped a full-rank g_proj. The pack
 * therefore carries kda_decay_gate_down_weight (fused f_a|g_a rows,
 * replicated) + kda_gate_up_weight instead of k3_gate_weight. */
#define SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT {kda_heads}
#define SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION {kda_dim}
#define SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION {kda_dim}
#define SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_KDA_CONV_KERNEL {conv}
#define SPARK_GLM5_NEXT_MODEL_KDA_CHUNK_TOKENS 64u
#define SPARK_GLM5_NEXT_MODEL_KDA_QK_L2NORM {l2norm}
#define SPARK_GLM5_NEXT_MODEL_KDA_FULL_RANK_GATE {full_rank}     /* per-head-per-channel decay (dt_bias [8192]) */
#define SPARK_GLM5_NEXT_MODEL_KDA_GATE_LOWER_BOUND {gate_lb}f
#define SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK {bottleneck}
#define SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL {conv}
#define SPARK_GLM5_NEXT_MODEL_KDA_STATE_ELEMENT_BYTES {state_bytes}
#define SPARK_GLM5_NEXT_MODEL_KDA_STATE_ELEMENTS_PER_HEAD \\
	(SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION * SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER \\
	(SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT * \\
	 SPARK_GLM5_NEXT_MODEL_KDA_STATE_ELEMENTS_PER_HEAD * \\
	 SPARK_GLM5_NEXT_MODEL_KDA_STATE_ELEMENT_BYTES)
/* The three bf16 short-conv windows (q|k|v) per KDA layer. */
#define SPARK_GLM5_NEXT_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER \\
	(3u * SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION * \\
	 SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL * 2u)
/* Forget gate: -5.0 * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)).
 * A_log is one f32 per head (64; no k3-style source-head slice). */
#define SPARK_GLM5_NEXT_MODEL_KDA_A_LOG_HEAD_COUNT SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT

/* -- MLA on DSA layers (glm52 donor) ----------------------------------------
 * DELTA 1: qk_rope_head_dim = 0. The query is nope-only (256/head), the KV
 * latent is the pure 512 lora (glm52: 576 = 512 + 64 rope), and there is NO
 * rope anywhere in the text stack (the reference passes
 * position_embeddings=None; the indexer never rotates either). Absorbed
 * scoring against the 512-wide latent with scale 256**-0.5; the kernel
 * wants a MLA_ROPE_DIM=0 instantiation of glm52's latent attention.
 * PORTING NOTE (coordinator correction 30e87ec): strip rope from BOTH the
 * MLA scoring AND the dsv4/glm52-donor indexer - config's
 * indexer_rope_interleave is NOT a porting dependency for this family; no
 * rope tables, no rope kernels, no positions input to attention at all. */
#define SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT {mla_heads}
#define SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION {q_a}
#define SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION {latent}
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION {nope}
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION {rope}
#define SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION {value_dim}
#define SPARK_GLM5_NEXT_MODEL_MLA_USE_NOPE {use_nope}
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_SCALE {qk_scale}f        /* 256 ** -0.5 */
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_HEAD_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + \\
	 SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_MLA_QK_HEAD_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION + \\
	 SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_MLA_KV_B_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * \\
	 (SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + \\
	  SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION))
#define SPARK_GLM5_NEXT_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT * \\
	 SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION)

/* -- DSA indexer (glm52 donor) + kpool compressor (dsv4 mechanism) ----------
 * Scores POOLS, not tokens: 4-token k-pools mixed by
 * softmax(gate_j + ape_j) over pool positions; per-head relu scores at
 * 128**-0.5; head weights weights_proj(x) * heads**-0.5 summed across
 * heads; select topk/kpool = 512 pools, expand to 2048 tokens plus the
 * incomplete tail (max 3), output width 2051, invalid = -1. NoPE: no rope
 * on index q or k. DELTA vs dsv4-0731: only ape[4,128] + gate[128,hidden]
 * exist (dsv4 carries ape+wkv+wgate+norm at 256 channels on ratio-4
 * layers). */
#define SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT {idx_heads}
#define SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION {idx_dim}
#define SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K {top_k}
#define SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL {kpool}
#define SPARK_GLM5_NEXT_MODEL_INDEX_POOL_SELECT_COUNT \\
	(SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL)
#define SPARK_GLM5_NEXT_MODEL_INDEX_TAIL_MAX (SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL - 1u)
#define SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH \\
	(SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K + SPARK_GLM5_NEXT_MODEL_INDEX_TAIL_MAX)
#define SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE {idx_softmax}f /* 128 ** -0.5 */
#define SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE {idx_head_scale}f /* 32 ** -0.5 */
#define SPARK_GLM5_NEXT_MODEL_INDEX_NORM_EPSILON {idx_eps}f  /* LayerNorm k_norm(w, b) */
/* The indexer cache packs [k(128) | gate(128) | valid(1)] per token. */
#define SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION \\
	(2u * SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION + 1u)

/* -- hyper-connections (dsv4 donor) -----------------------------------------
 * mHC: fn [24, 4*hidden] over the UNWEIGHTED-RMSNorm'd flattened streams;
 * pre = sigmoid(w*s0 + b0) + eps; post = 2*sigmoid(w*s1 + b1);
 * comb = softmax + eps, one column norm, then 19 row/col norm pairs
 * (sinkhorn 20 total). Streams init to the embedding expanded across 4;
 * the FINAL head collapse is an UNWEIGHTED MEAN (dsv4 uses a weighted
 * hc_head) then RMSNorm. The MTP layer carries no hc_* tensors. */
#define SPARK_GLM5_NEXT_MODEL_HC_MULT {hc_mult}
#define SPARK_GLM5_NEXT_MODEL_HC_SINKHORN_ITERATIONS {hc_iter}
#define SPARK_GLM5_NEXT_MODEL_HC_EPSILON {hc_eps}f
#define SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION \\
	((2u + SPARK_GLM5_NEXT_MODEL_HC_MULT) * SPARK_GLM5_NEXT_MODEL_HC_MULT)
#define SPARK_GLM5_NEXT_MODEL_HC_FN_COLUMNS \\
	(SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT {hc_scales}

/* -- MoE (glm52 donor) -------------------------------------------------------
 * Sigmoid router with frozen e_score_correction bias, noaux_tc at
 * n_group = 1, norm_topk_prob, routed scaling 2.5 - the glm52 config
 * verbatim except 288 experts. Router logits in fp32. Dense MLP (int
 * 12288) on the first 3 layers; shared expert intermediate is 2048 (one
 * shared expert). */
#define SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT {experts}
#define SPARK_GLM5_NEXT_MODEL_MOE_TOP_K {topk}
#define SPARK_GLM5_NEXT_MODEL_MOE_SHARED_EXPERT_COUNT {shared}
#define SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION {inter}
#define SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION {dense_inter}
#define SPARK_GLM5_NEXT_MODEL_FIRST_DENSE_LAYER_COUNT {first_dense}
#define SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER {first_dense}
#define SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_SCALING_FACTOR {scaling}f
#define SPARK_GLM5_NEXT_MODEL_MOE_NORM_TOPK_PROB {norm_topk}
#define SPARK_GLM5_NEXT_MODEL_MOE_W1_COMPONENT_COUNT {w1c}

/* -- quantisation ------------------------------------------------------------
 * FP8 e4m3 dynamic with [128,128] blocks on: routed experts, shared
 * experts, dense MLP, and the MLA q_a/q_b/kv_a/o projections. KDA tensors,
 * indexer tensors, hc_*, router, norms, embed/lm_head stay BF16/F32 (the
 * contract's precision.fp8_quantized_patterns is the exact set). */
#define SPARK_GLM5_NEXT_MODEL_FP8_SCALE_BLOCK {fp8_block}

/* -- KV geometry -------------------------------------------------------------
 * DSA slots hold the pure 512-wide latent (1024 B at bf16; no rope
 * segment). KDA state is fp32 64x128x128 = 4 MiB per layer (~140 MiB
 * across 34 layers). The indexer cache holds the packed 257-float row per
 * token per DSA layer. Page policy matches glm52. */
#define SPARK_GLM5_NEXT_MODEL_KV_BITS {kv_bits}
#define SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS {page}
#define SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES \\
	((SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION * SPARK_GLM5_NEXT_MODEL_KV_BITS) / 8u)

/* -- derived ---------------------------------------------------------------- */
/* glm52-lineage aliases the module host code speaks (the module skeleton
 * descends from glm52's; these name the same geometry). */
#define SPARK_GLM5_NEXT_MODEL_HEAD_COUNT \\
	SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT
#define SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION \\
	SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION
#define SPARK_GLM5_NEXT_MODEL_QUERY_A_DIMENSION \\
	SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION
#define SPARK_GLM5_NEXT_MODEL_QUERY_B_DIMENSION \\
	SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION
#define SPARK_GLM5_NEXT_MODEL_VALUE_HEAD_DIMENSION \\
	SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION
#define SPARK_GLM5_NEXT_MODEL_ROPE_DIMENSION \\
	SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION
#define SPARK_GLM5_NEXT_MODEL_CACHE_TOKEN_ELEMENTS \\
	SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION
#define SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES {bf16}
#define SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_COUNT \\
	SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT
#define SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_DIMENSION \\
	SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION
#define SPARK_GLM5_NEXT_MODEL_DSA_INDEX_QUERY_DIMENSION \\
	SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION
#define SPARK_GLM5_NEXT_MODEL_DSA_SELECTED_TOKEN_COUNT \\
	SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K
#define SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_GATE_UP_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION * \\
	 SPARK_GLM5_NEXT_MODEL_MOE_W1_COMPONENT_COUNT)
#define SPARK_GLM5_NEXT_MODEL_ROUTED_LAYERS \\
	(SPARK_GLM5_NEXT_MODEL_LAYER_COUNT - SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER)
#define SPARK_GLM5_NEXT_MODEL_GATE_UP_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION * \\
	 SPARK_GLM5_NEXT_MODEL_MOE_W1_COMPONENT_COUNT)
#define SPARK_GLM5_NEXT_MODEL_WEIGHT_LAYER_COUNT \\
	(SPARK_GLM5_NEXT_MODEL_LAYER_COUNT + 1u)  /* + the MTP layer */

/* -- compile-time sanity ----------------------------------------------------- */
#if SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION != 0u
#error glm5_next is the rope-0 MLA instantiation; a nonzero rope dim belongs to another family
#endif
#if SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION % 128u != 0u
#error kda qkv rows must stay whole-head at the TP16 slice
#endif
#if (SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT % 16u) != 0u || \\
	(SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT % 16u) != 0u || \\
	(SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT % 16u) != 0u
#error glm5_next assumes TP16: every head count must divide by 16
#endif
#if (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT % 16u) != 0u
#error glm5_next assumes TP16: the expert count must divide into 16 ranks
#endif
#if SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH != 2051u
#error pool expansion width changed; update the indexer consumers
#endif

#endif
"""


def render_qwen4_flash(c: dict) -> str:
    model, hybrid, linear, attn = c["model"], c["hybrid_attention"], c["linear_attn"], c["attention"]
    moe, hc, index, ple = c["moe"], c["hyper_connection"], c["indexer"], c["ple"]
    policy = FAMILY_POLICY["qwen4_flash"]
    require(hybrid["linear_layer_count"] + hybrid["full_layer_count"] == model["layer_count"],
            "qwen4_flash hybrid layer partition must cover the stack")
    require(moe["routed_expert_count"] == model["routed_expert_count"], "qwen4_flash expert count disagrees between sections")

    hidden, layers, vocab = u(model["hidden_dimension"]), u(model["layer_count"]), u(model["vocabulary_size"])
    heads, kv_heads, head_dim = u(model["attention_head_count"]), u(model["kv_head_count"]), u(model["head_dimension"])
    gdn_key_heads, gdn_value_heads = u(linear["key_head_count"]), u(linear["value_head_count"])
    max_ctx = u(model["maximum_context_tokens"])
    eps = f"{model['rms_norm_epsilon']:.0e}"
    mtp = u(model["mtp_layer_count"])
    period, phase = u(hybrid["period"]), u(hybrid["full_phase"])
    gdn_layers, full_layers = u(hybrid["linear_layer_count"]), u(hybrid["full_layer_count"])
    gdn_key_dim, gdn_value_dim = u(linear["key_dimension"]), u(linear["value_dimension"])
    conv = u(linear["short_conv_kernel"])
    rope_dim = u(attn["rope_dimension"])
    theta = repr(float(attn["rope_theta"]))
    experts = u(moe["routed_expert_count"])
    topk = u(moe["experts_per_token"])
    shared = u(moe["shared_expert_count"])
    inter = u(moe["expert_intermediate_dimension"])
    shared_inter = u(moe["shared_expert_intermediate_dimension"])
    mxfp4 = u(policy["mxfp4_group_size"])
    fp8_block = u(policy["fp8_block"])
    swiglu = repr(float(policy["swiglu_limit"]))
    hc_streams = u(hc["stream_count"])
    hc_lowrank = u(hc["lowrank_dimension"])
    idx_heads = u(index["head_count"])
    idx_kv_heads = u(index["kv_head_count"])
    idx_dim = u(index["head_dimension"])
    idx_budget = u(index["budget"])
    idx_ratio = u(index["compress_ratio"])
    ple_layer = u(ple["layer_index_weights"])
    ple_ngram = u(ple["ngram_size"])
    ple_heads = u(ple["heads_per_ngram"])
    ple_shards = u(ple["shard_count"])
    ple_conv = u(ple["conv_kernel"])
    chunk = u(policy["gdn_chunk_tokens"])

    return f"""/* Generated by tools/gen_geometry_header.py from
 * model_contracts/qwen4_flash_authoritative.json - do not edit by hand.
 * Cutover 2026-08-27 (W4 redundancy lane): diff-reviewed against the
 * hand-written original; content-identical except this banner (values,
 * prose and macro set unchanged). */
// Qwen 3.8 Flash model geometry for the host tiers, pinned against the
// qwen4_flash authoritative contract (model_contracts/
// qwen4_flash_authoritative.json), which is in turn digest-frozen against
// the warm checkpoint config.json at revision f5d0827. The firmware config
// remains the source of truth; tests/test_qwen4_flash_model_header.py holds
// header and contract in lockstep, same discipline as qwen38_max/qwen38_27b.
//
// Sibling of qwen38_max: same 3:1 linear:full hybrid period and the same
// 512-expert top-10 MoE shape. Checkpoint-verified deltas vs the max
// sibling (see the contract's tensor_census): hyper-connection residual
// mixing on every sublayer and a final mixer instead of a plain final norm,
// an attention indexer on the 12 full-attention layers, and a PLE n-gram
// embedding block on layer 1 (weights; config ple_layer_ids says 2 - the
// tensors win). Those constants are pinned here too so the module and packer
// consume one geometry.
#ifndef SPARKPIPE_SPARK_QWEN4_FLASH_MODEL_H
#define SPARKPIPE_SPARK_QWEN4_FLASH_MODEL_H

#define SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION {hidden}
#define SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT {layers}
#define SPARK_QWEN4_FLASH_MODEL_VOCAB_COUNT {vocab}
#define SPARK_QWEN4_FLASH_MODEL_ATTENTION_HEAD_COUNT {heads}
#define SPARK_QWEN4_FLASH_MODEL_KV_HEAD_COUNT {kv_heads}
#define SPARK_QWEN4_FLASH_MODEL_HEAD_DIMENSION {head_dim}
#define SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT {gdn_key_heads}
#define SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT {gdn_value_heads}

// Wider geometry vocabulary for the resident decode stage module.
#define SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT SPARK_QWEN4_FLASH_MODEL_VOCAB_COUNT
#define SPARK_QWEN4_FLASH_MODEL_MAXIMUM_CONTEXT_TOKENS {max_ctx}
#define SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON {eps}f
#define SPARK_QWEN4_FLASH_MODEL_MTP_LAYER_COUNT {mtp}

#define SPARK_QWEN4_FLASH_MODEL_ATTENTION_PERIOD {period}
#define SPARK_QWEN4_FLASH_MODEL_FULL_ATTENTION_PHASE {phase}
#define SPARK_QWEN4_FLASH_MODEL_LAYER_IS_GDN(layer_index) \\
	(((layer_index) % SPARK_QWEN4_FLASH_MODEL_ATTENTION_PERIOD) != SPARK_QWEN4_FLASH_MODEL_FULL_ATTENTION_PHASE)
#define SPARK_QWEN4_FLASH_MODEL_GDN_LAYER_COUNT {gdn_layers}
#define SPARK_QWEN4_FLASH_MODEL_FULL_ATTENTION_LAYER_COUNT {full_layers}

// Linear attention ("gdn" slot in the module, GatedDeltaNet-shaped): the
// checkpoint's linear_attn.in_proj_* map one-to-one onto the qwen38_max GDN
// tensor kinds - in_proj_qkv [2*QK+V, H], in_proj_z (gate) [V, H],
// in_proj_a (decay) / in_proj_b (beta) [value_heads, H], norm [128],
// conv1d [2*QK+V, 1, 4], A_log / dt_bias [value_heads], out_proj [H, V].
// The grouped-value ratio is 3 (48 value heads over 16 key heads), NOT the
// max sibling's 8; kernels must derive it from these macros.
#define SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_KEY_DIMENSION {gdn_key_dim}
#define SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_VALUE_DIMENSION {gdn_value_dim}
#define SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD \\
	(SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT)
#define SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL {conv}
#define SPARK_QWEN4_FLASH_MODEL_GDN_QK_DIMENSION \\
	(SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_KEY_DIMENSION)
#define SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION \\
	(SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_VALUE_DIMENSION)
#define SPARK_QWEN4_FLASH_MODEL_GDN_CONV_CHANNELS \\
	((2u * SPARK_QWEN4_FLASH_MODEL_GDN_QK_DIMENSION) + SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION)
#define SPARK_QWEN4_FLASH_MODEL_GDN_CONV_TAIL_COLUMNS (SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL - 1u)
#define SPARK_QWEN4_FLASH_MODEL_GDN_CHUNK_TOKENS {chunk}

// Gated full attention: q_proj carries [2*Q, H] (query then sigmoid output
// gate, output_gate_type sigmoid), k/v [512, H] (2 heads x 256), partial
// rotary 0.25 -> 64 rope dimensions per head, mrope sections [11,11,10]
// degenerate to plain rope for text-only decode (same as the max contract).
#define SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT SPARK_QWEN4_FLASH_MODEL_ATTENTION_HEAD_COUNT
#define SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT SPARK_QWEN4_FLASH_MODEL_KV_HEAD_COUNT
#define SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION SPARK_QWEN4_FLASH_MODEL_HEAD_DIMENSION
#define SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION {rope_dim}
#define SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_THETA {theta}f
#define SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_DIMENSION \\
	(SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_QWEN4_FLASH_MODEL_ATTN_KV_DIMENSION \\
	(SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_QWEN4_FLASH_MODEL_ATTN_CACHE_TOKEN_ELEMENTS \\
	(2u * SPARK_QWEN4_FLASH_MODEL_ATTN_KV_DIMENSION)

// Routed MoE, pinned against config.json at the contract revision: same
// expert shape family as qwen38_max with the Flash intermediate (640).
// Source experts are BF16 fused gate_up [512, 1280, H]; the packer splits
// w1/w3 and quantizes to the format-6 MX FP8 codec (E4M3 payload, E8M0
// scale, 128 blocks), matching the qwen38_27b serving format.
#define SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT {experts}
#define SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN {topk}
#define SPARK_QWEN4_FLASH_MODEL_SHARED_EXPERT_COUNT {shared}
#define SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION {inter}
#define SPARK_QWEN4_FLASH_MODEL_SHARED_EXPERT_INTERMEDIATE_DIMENSION {shared_inter}
#define SPARK_QWEN4_FLASH_MODEL_MXFP4_GROUP_SIZE {mxfp4}
#define SPARK_QWEN4_FLASH_MODEL_FP8_BLOCK {fp8_block}
#define SPARK_QWEN4_FLASH_MODEL_SWIGLU_LIMIT {swiglu}f

// Hyper-connection residual geometry (hc_count 4, low-rank 320): the
// residual stream is 4 x hidden wide; every attention and MoE sublayer
// carries a mixer (input_mix up [4H, 320] / down [320, 4H], block_inject
// [4, 4H], hc_norm [4H]), and the stack ends in a global
// hyper_connection_mixer of the same shape instead of a plain final norm.
#define SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT {hc_streams}
#define SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION {hc_lowrank}
#define SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH \\
	(SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION)

// Attention indexer on every full-attention layer (DSV4-flash family):
// 4 query heads + 1 kv head at 128, top-k budget 2048, compression 4.
#define SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT {idx_heads}
#define SPARK_QWEN4_FLASH_MODEL_INDEXER_KV_HEAD_COUNT {idx_kv_heads}
#define SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION {idx_dim}
#define SPARK_QWEN4_FLASH_MODEL_INDEXER_BUDGET {idx_budget}
#define SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO {idx_ratio}

// PLE n-gram embedding block: tensors live on layer 1 (weights truth; the
// config's ple_layer_ids [2] is drifted - see the contract census). 3-gram,
// 8 heads per n-gram, embedding dim = hidden, conv kernel 4, 128 shards.
#define SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX {ple_layer}
#define SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_SIZE {ple_ngram}
#define SPARK_QWEN4_FLASH_MODEL_PLE_HEADS_PER_NGRAM {ple_heads}
#define SPARK_QWEN4_FLASH_MODEL_PLE_SHARD_COUNT {ple_shards}
#define SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION
#define SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL {ple_conv}

#define SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES 2u
#define SPARK_QWEN4_FLASH_MODEL_HIDDEN_BF16_BYTES \\
	(SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES)

#endif
"""


RENDERERS = {
    "qwen38_27b": render_qwen38_27b,
    "glm5_next": render_glm5_next,
    "qwen4_flash": render_qwen4_flash,
}


def render_header(family: str, contract: dict) -> str:
    return RENDERERS[family](contract)


def emit_adapter_constants(family: str, contract: dict) -> str:
    """The serving-adapter descriptor constants blob for the family.

    Emitted next to the geometry header (model-families/<family>/...) so the
    adapter's pasted constants block becomes an include; the cutover edit in
    modules/ belongs to the W2 consolidation lane (their write set).
    """
    if family != "qwen38_27b":
        raise ValueError(f"adapter-constants emission not yet modelled for {family}")
    model = contract["model"]
    revision = contract["revision"]
    require(contract["model_id"] == "Qwen/Qwen3.8-27B", "contract model id moved")
    stage_layers = ", ".join(["64u"] * 4 + ["0u"] * 12)
    stage_layers_tp1 = ", ".join(["64u"] * 1 + ["0u"] * 15)
    max_ctx = u(model["maximum_context_tokens"])
    return f"""/* Serving-adapter descriptor constants for the qwen38_27b family.
 * Generated by tools/gen_geometry_header.py from
 * model_contracts/qwen38_27b_authoritative.json - do not edit by hand.
 * Diff-reviewed against the constants pasted in
 * modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c;
 * the include-cutover edit in modules/ is the W2 consolidation lane's. */
#ifndef SPARKPIPE_SPARK_QWEN38_27B_SERVING_CONSTANTS_H
#define SPARKPIPE_SPARK_QWEN38_27B_SERVING_CONSTANTS_H

/* Serving topology build knob, overridable on the compile line:
 *   4 (default) = shipped TP4 whole-stack build; 1 = TP1 full-width. */
#ifndef SPARK_QWEN38_27B_SERVING_TP_DEGREE
#define SPARK_QWEN38_27B_SERVING_TP_DEGREE 4u
#endif
#define SPARK_QWEN38_27B_SERVING_TP (SPARK_QWEN38_27B_SERVING_TP_DEGREE >= 1u)

#define SPARK_QWEN38_27B_SERVING_MODEL_ID "Qwen/Qwen3.8-27B"
#define SPARK_QWEN38_27B_SERVING_MODEL_REVISION "{revision}"
#define SPARK_QWEN38_27B_SERVING_DRIVER_MODEL_ID \\
	"alibaba.qwen3.8-27b.resident-decode-stage-firmware"
#define SPARK_QWEN38_27B_SERVING_STAGE_NAME "qwen38_27b_resident_decode_stage"
#define SPARK_QWEN38_27B_SERVING_TARGET \\
	"cuda.sm121.qwen38_27b.resident_decode_stage.bf16"
#define SPARK_QWEN38_27B_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_QWEN38_27B_SERVING_MAX_SEQUENCE_POSITIONS_CAP {max_ctx}

#if SPARK_QWEN38_27B_SERVING_TP_DEGREE == 1u
#define SPARK_QWEN38_27B_SERVING_ADAPTER_ID "spark.qwen38_27b.serving-adapter.tp1.v1"
#define SPARK_QWEN38_27B_SERVING_STAGE_COUNT 1u
#define SPARK_QWEN38_27B_SERVING_STAGE_LAYER_COUNTS {{{stage_layers_tp1}}}
#else
#define SPARK_QWEN38_27B_SERVING_ADAPTER_ID "spark.qwen38_27b.serving-adapter.tp4.v1"
#define SPARK_QWEN38_27B_SERVING_STAGE_COUNT 4u
#define SPARK_QWEN38_27B_SERVING_STAGE_LAYER_COUNTS {{{stage_layers}}}
#endif

#endif
"""


def write_or_check(relative: str, content: str, check_only: bool) -> bool:
    target = ROOT / relative
    if check_only:
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != content:
            print(f"DRIFT {relative} (regenerated output differs from the checked-in file)")
            return False
        print(f"ok    {relative} (byte-identical)")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"wrote {relative}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--check", action="store_true",
                        help="verify the regenerated file matches the checked-in one byte-for-byte")
    parser.add_argument("--emit-adapter-constants", action="store_true",
                        help="emit the family serving-adapter descriptor constants header")
    arguments = parser.parse_args()

    family = FAMILIES[arguments.family]
    contract = load_contract(family["contract"])
    if arguments.emit_adapter_constants:
        relative = ADAPTER_CONSTANTS[arguments.family]
        content = emit_adapter_constants(arguments.family, contract)
    else:
        relative = family["header"]
        content = render_header(arguments.family, contract)
    return 0 if write_or_check(relative, content, arguments.check) else 1


if __name__ == "__main__":
    sys.exit(main())
