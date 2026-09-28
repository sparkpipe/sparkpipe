#!/usr/bin/env python3
"""Bind the ling geometry header to the authoritative contract.

The contract (model_contracts/ling_authoritative.json) pins geometry to
the publisher config (inclusionAI/Ling-3.0-flash @
e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3). This test binds
model-families/ling/include/sparkpipe/spark_ling_model.h to that contract.
Run: python3 tests/test_ling_model_header.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from c_macro_values import MacroProbeError, c_macro_values, value_matches

REPOSITORY = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORIES = [REPOSITORY / "model-families/ling/include",
                       REPOSITORY / "model-families/common/include"]
CONTRACT = REPOSITORY / "model_contracts/ling_authoritative.json"

BINDINGS = {
    ("model", "hidden_dimension"): "SPARK_LING_MODEL_HIDDEN_DIMENSION",
    ("model", "layer_count"): "SPARK_LING_MODEL_LAYER_COUNT",
    ("model", "vocabulary_size"): "SPARK_LING_MODEL_OUTPUT_VOCAB_COUNT",
    ("model", "mtp_layer_index"): "SPARK_LING_MODEL_MTP_LAYER_INDEX",
    ("model", "maximum_context_tokens"): "SPARK_LING_MODEL_MAXIMUM_CONTEXT_TOKENS",
    ("model", "rms_norm_epsilon"): "SPARK_LING_MODEL_RMS_NORM_EPSILON",
    ("model", "end_of_text_token_id"): "SPARK_LING_MODEL_END_OF_TEXT_TOKEN_ID",
    ("model", "pad_token_id"): "SPARK_LING_MODEL_PAD_TOKEN_ID",
    ("hybrid_attention", "period"): "SPARK_LING_MODEL_ATTENTION_PERIOD",
    ("hybrid_attention", "mla_phase"): "SPARK_LING_MODEL_GLOBAL_ATTENTION_PHASE",
    ("hybrid_attention", "mla_layer_count"): "SPARK_LING_MODEL_MLA_LAYER_COUNT",
    ("hybrid_attention", "kda_layer_count"): "SPARK_LING_MODEL_KDA_LAYER_COUNT",
    ("mla", "head_count"): "SPARK_LING_MODEL_MLA_HEAD_COUNT",
    ("mla", "latent_dimension"): "SPARK_LING_MODEL_MLA_LATENT_DIMENSION",
    ("mla", "qk_nope_head_dimension"): "SPARK_LING_MODEL_MLA_QK_NOPE_HEAD_DIMENSION",
    ("mla", "qk_rope_head_dimension"): "SPARK_LING_MODEL_MLA_QK_ROPE_HEAD_DIMENSION",
    ("mla", "value_head_dimension"): "SPARK_LING_MODEL_MLA_VALUE_HEAD_DIMENSION",
    ("mla", "rope_theta"): "SPARK_LING_MODEL_MLA_ROPE_THETA",
    ("mla", "qk_scale"): "SPARK_LING_MODEL_MLA_QK_SCALE",
    ("kda", "head_count"): "SPARK_LING_MODEL_KDA_HEAD_COUNT",
    ("kda", "key_dimension"): "SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION",
    ("kda", "value_dimension"): "SPARK_LING_MODEL_KDA_HEAD_VALUE_DIMENSION",
    ("kda", "short_conv_kernel"): "SPARK_LING_MODEL_KDA_CONV_KERNEL",
    ("kda", "chunk_tokens"): "SPARK_LING_MODEL_KDA_CHUNK_TOKENS",
    ("kda", "gate_lower_bound"): "SPARK_LING_MODEL_KDA_GATE_LOWER_BOUND",
    ("moe", "routed_expert_count"): "SPARK_LING_MODEL_MOE_EXPERT_COUNT",
    ("moe", "experts_per_token"): "SPARK_LING_MODEL_MOE_TOP_K",
    ("moe", "shared_expert_count"): "SPARK_LING_MODEL_MOE_SHARED_EXPERT_COUNT",
    ("moe", "expert_intermediate_dimension"): "SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION",
    ("moe", "dense_intermediate_dimension"): "SPARK_LING_MODEL_DENSE_INTERMEDIATE_DIMENSION",
    ("moe", "first_dense_layer_count"): "SPARK_LING_MODEL_FIRST_DENSE_LAYER_COUNT",
    ("moe", "routed_scaling_factor"): "SPARK_LING_MODEL_MOE_ROUTED_SCALING_FACTOR",
    ("moe", "router_group_count"): "SPARK_LING_MODEL_MOE_ROUTER_GROUP_COUNT",
    ("moe", "router_top_groups"): "SPARK_LING_MODEL_MOE_ROUTER_TOP_GROUPS",
}


def composed_expectations(contract: dict) -> dict[str, int]:
    mla = contract["mla"]
    kda = contract["kda"]
    model = contract["model"]
    prefix = "SPARK_LING_MODEL_"
    return {
        prefix + "MLA_QUERY_DIMENSION":
            mla["head_count"] * (mla["qk_nope_head_dimension"] + mla["qk_rope_head_dimension"]),
        prefix + "MLA_KV_A_DIMENSION": mla["latent_dimension"] + mla["qk_rope_head_dimension"],
        prefix + "MLA_KV_B_DIMENSION":
            mla["head_count"] * (mla["qk_nope_head_dimension"] + mla["value_head_dimension"]),
        prefix + "MLA_ATTENTION_PROJECTION_DIMENSION": mla["head_count"] * mla["value_head_dimension"],
        prefix + "KDA_QKV_DIMENSION": kda["head_count"] * kda["key_dimension"],
        prefix + "KDA_VALUE_DIMENSION": kda["head_count"] * kda["value_dimension"],
        prefix + "KDA_STATE_BYTES_PER_LAYER":
            kda["head_count"] * kda["key_dimension"] * kda["value_dimension"] * 4,
        prefix + "KV_SLOT_BYTES": (mla["latent_dimension"] + mla["qk_rope_head_dimension"]) * 2,
        prefix + "WEIGHT_LAYER_COUNT": model["layer_count"] + 1,
    }


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    prefix = "SPARK_LING_MODEL_"
    layer_count = contract["model"]["layer_count"]
    composed = composed_expectations(contract)
    mla_flags = [prefix + f"LAYER_IS_MLA({layer}u)" for layer in range(layer_count)]
    kda_flags = [prefix + f"LAYER_IS_KDA({layer}u)" for layer in range(layer_count)]
    try:
        values = c_macro_values(INCLUDE_DIRECTORIES, "sparkpipe/spark_ling_model.h",
                                list(BINDINGS.values()) + list(composed) + mla_flags + kda_flags,
                                ("SPARK_BATCH_BUCKET=1024u",))
    except MacroProbeError as error:
        print(f"FAILED {error}")
        return 1
    failures = []
    for (section, key), name in BINDINGS.items():
        expected = contract[section][key]
        if not value_matches(expected, values[name]):
            failures.append(f"{name}: header {values[name]!r} contract {expected!r}")
    for name, expected in composed.items():
        if not value_matches(expected, values[name]):
            failures.append(f"composed {name}: header {values[name]!r} expected {expected!r}")
    period = values[prefix + "ATTENTION_PERIOD"]
    phase = values[prefix + "GLOBAL_ATTENTION_PHASE"]
    mla_layers = [layer for layer, flag in enumerate(mla_flags) if values[flag] != 0.0]
    kda_layers = [layer for layer, flag in enumerate(kda_flags) if values[flag] != 0.0]
    if mla_layers != [layer for layer in range(layer_count) if layer % period == phase]:
        failures.append(f"LAYER_IS_MLA marks layers {mla_layers}")
    if sorted(mla_layers + kda_layers) != list(range(layer_count)):
        failures.append("every layer must be exactly one of MLA or KDA")
    if len(mla_layers) != values[prefix + "MLA_LAYER_COUNT"] or \
            len(kda_layers) != values[prefix + "KDA_LAYER_COUNT"]:
        failures.append(f"{len(mla_layers)} MLA and {len(kda_layers)} KDA layers disagree with the "
                        f"header counts {values[prefix + 'MLA_LAYER_COUNT']} and "
                        f"{values[prefix + 'KDA_LAYER_COUNT']}")
    if values[prefix + "MLA_HEAD_COUNT"] % 16 or values[prefix + "MOE_EXPERT_COUNT"] % 16:
        failures.append("TP16: head and expert counts must divide by 16")
    if not values[prefix + "MOE_ROUTER_TOP_GROUPS"] < values[prefix + "MOE_ROUTER_GROUP_COUNT"]:
        failures.append("router top groups must be a strict subset of groups")
    if values[prefix + "MOE_INTERMEDIATE_DIMENSION"] % 128 or values[prefix + "HIDDEN_DIMENSION"] % 128:
        failures.append("expert geometry must tile 128-block scale groups")
    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        print(f"FAILED {len(failures)} binding(s)")
        return 1
    print(f"PASS ling header matches the authoritative contract ({len(BINDINGS)} bindings + "
          f"{len(composed)} composed + {layer_count} layer kinds, evaluated by the C compiler)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
