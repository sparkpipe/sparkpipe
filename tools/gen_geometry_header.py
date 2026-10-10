#!/usr/bin/env python3
"""Geometry-header generator: the recipe compiler v0 (W4 redundancy lane).

model_contracts/<family>_authoritative.json -> the family geometry header
(model-families/<family>/include/sparkpipe/spark_<family>_model.h). Every geometry number in
the emitted header is read from the contract (strict indexing - a missing key
is an error, never a default); prose and derived-macro structure are owned by
this file's family templates, the same split the dsv4/k3 contract generators
use. Provenance for constants that are module/deployment policy rather than
model geometry is recorded in FAMILY_POLICY below.

Proof discipline (docs/HOUSECLEANING_PLAN.md W4.4): where a hand-written
original exists, --check must reproduce it byte-identical before cutover.
The family header is byte-identical to the output; tests/test_gen_geometry_header.py
runs --check for every output.

Usage:
    python3 tools/gen_geometry_header.py --family glm5_next [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "glm5_next": {
        "contract": "model_contracts/glm53_flash_authoritative.json",
        "header": "model-families/glm5_next/include/sparkpipe/spark_glm5_next_model.h",
    },
}

# Constants that are module/deployment policy, not checkpoint geometry; the
# contract does not carry them. Each entry names its owner so the split stays
# auditable. Values land in the emitted header verbatim.
FAMILY_POLICY = {
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
        "replay_rows_max": 8,
        "miss_pack_stride": 512,
        "miss_ring_capacity": 1024,
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
    replay_rows = u(policy["replay_rows_max"])
    miss_stride = u(policy["miss_pack_stride"])
    miss_ring = u(policy["miss_ring_capacity"])
    require(policy["miss_pack_stride"] >= moe["routed_expert_count"],
            "glm5_next miss packing stride must cover the expert count")

    return f"""#ifndef SPARKPIPE_SPARK_GLM5_NEXT_MODEL_H
#define SPARKPIPE_SPARK_GLM5_NEXT_MODEL_H

#include <stdint.h>

#define SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION {hidden}
#define SPARK_GLM5_NEXT_MODEL_LAYER_COUNT {layers}
#define SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX {mtp_index}
#define SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT {vocab}
#define SPARK_GLM5_NEXT_REPLAY_ROWS_MAX {replay_rows}
#define SPARK_GLM5_NEXT_MODEL_MAXIMUM_CONTEXT_TOKENS {max_ctx}
#define SPARK_GLM5_NEXT_MODEL_RMS_NORM_EPSILON {eps}f
#define SPARK_GLM5_NEXT_MODEL_SWIGLU_LIMIT {swiglu}f
#define SPARK_GLM5_NEXT_MODEL_END_OF_TEXT_TOKEN_ID {eos}
#define SPARK_GLM5_NEXT_MODEL_USER_TOKEN_ID {user}
#define SPARK_GLM5_NEXT_MODEL_OBSERVATION_TOKEN_ID {obs}
#define SPARK_GLM5_NEXT_MODEL_PAD_TOKEN_ID {pad}

#define SPARK_GLM5_NEXT_MODEL_KV_POOL_TOKENS {pool}
#define SPARK_GLM5_NEXT_MODEL_RESTRICTED_VOCAB_COUNT {restricted}
#define SPARK_GLM5_NEXT_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH {prefill}

#define SPARK_GLM5_NEXT_MODEL_ATTENTION_PERIOD {period}
#define SPARK_GLM5_NEXT_MODEL_GLOBAL_ATTENTION_PHASE {phase}
#define SPARK_GLM5_NEXT_MODEL_KDA_LAYER_COUNT {kda_layers}
#define SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT {dsa_layers}
#define SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer_index) \\
	(((layer_index) % SPARK_GLM5_NEXT_MODEL_ATTENTION_PERIOD) != \\
	 SPARK_GLM5_NEXT_MODEL_GLOBAL_ATTENTION_PHASE)

#define SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT {kda_heads}
#define SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION {kda_dim}
#define SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION {kda_dim}
#define SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION \\
	(SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_KDA_CONV_KERNEL {conv}
#define SPARK_GLM5_NEXT_MODEL_KDA_CHUNK_TOKENS 64u
#define SPARK_GLM5_NEXT_MODEL_KDA_QK_L2NORM {l2norm}
#define SPARK_GLM5_NEXT_MODEL_KDA_FULL_RANK_GATE {full_rank}
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
#define SPARK_GLM5_NEXT_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER \\
	(3u * SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION * \\
	 SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL * 2u)
#define SPARK_GLM5_NEXT_MODEL_KDA_A_LOG_HEAD_COUNT SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT

#define SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT {mla_heads}
#define SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION {q_a}
#define SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION {latent}
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION {nope}
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION {rope}
#define SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION {value_dim}
#define SPARK_GLM5_NEXT_MODEL_MLA_USE_NOPE {use_nope}
#define SPARK_GLM5_NEXT_MODEL_MLA_QK_SCALE {qk_scale}f
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
#define SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE {idx_softmax}f
#define SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE {idx_head_scale}f
#define SPARK_GLM5_NEXT_MODEL_INDEX_NORM_EPSILON {idx_eps}f
#define SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION \\
	(2u * SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION + 1u)

#define SPARK_GLM5_NEXT_MODEL_HC_MULT {hc_mult}
#define SPARK_GLM5_NEXT_MODEL_HC_SINKHORN_ITERATIONS {hc_iter}
#define SPARK_GLM5_NEXT_MODEL_HC_EPSILON {hc_eps}f
#define SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION \\
	((2u + SPARK_GLM5_NEXT_MODEL_HC_MULT) * SPARK_GLM5_NEXT_MODEL_HC_MULT)
#define SPARK_GLM5_NEXT_MODEL_HC_FN_COLUMNS \\
	(SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION)
#define SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT {hc_scales}

#define SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT {experts}
#define SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE {miss_stride}
#define SPARK_GLM5_NEXT_MODEL_MISS_RING_CAPACITY {miss_ring}
#define SPARK_GLM5_NEXT_MODEL_MISS_RING_BYTES \\
    ((2u + SPARK_GLM5_NEXT_MODEL_MISS_RING_CAPACITY + 2u) * 4u)
#define SPARK_GLM5_NEXT_MODEL_MISS_EPOCH_WORD_U64 \\
    ((2u + SPARK_GLM5_NEXT_MODEL_MISS_RING_CAPACITY) / 2u)
#if SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE < SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT
#error "miss packing stride must exceed the expert count"
#endif
#define SPARK_GLM5_NEXT_MODEL_MOE_TOP_K {topk}
#define SPARK_GLM5_NEXT_MODEL_MOE_SHARED_EXPERT_COUNT {shared}
#define SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION {inter}
#define SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION {dense_inter}
#define SPARK_GLM5_NEXT_MODEL_FIRST_DENSE_LAYER_COUNT {first_dense}
#define SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER {first_dense}
#define SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_SCALING_FACTOR {scaling}f
#define SPARK_GLM5_NEXT_MODEL_MOE_NORM_TOPK_PROB {norm_topk}
#define SPARK_GLM5_NEXT_MODEL_MOE_W1_COMPONENT_COUNT {w1c}

#define SPARK_GLM5_NEXT_MODEL_FP8_SCALE_BLOCK {fp8_block}

#define SPARK_GLM5_NEXT_MODEL_KV_BITS {kv_bits}
#define SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS {page}
#define SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES \\
	((SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION * SPARK_GLM5_NEXT_MODEL_KV_BITS) / 8u)

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
	(SPARK_GLM5_NEXT_MODEL_LAYER_COUNT + 1u)

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


RENDERERS = {
    "glm5_next": render_glm5_next,
}


def render_header(family: str, contract: dict) -> str:
    return RENDERERS[family](contract)


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
    arguments = parser.parse_args()

    family = FAMILIES[arguments.family]
    contract = load_contract(family["contract"])
    relative = family["header"]
    content = render_header(arguments.family, contract)
    return 0 if write_or_check(relative, content, arguments.check) else 1


if __name__ == "__main__":
    sys.exit(main())
