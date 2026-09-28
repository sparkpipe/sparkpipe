#!/usr/bin/env python3
"""Bind the qwen4_flash geometry header to the authoritative contract.

The contract (model_contracts/qwen4_flash_authoritative.json) is digest-
frozen against the warm checkpoint by tools/qwen4_flash_verify_source.py
(verify mode, run on a spark node); this test binds
model-families/qwen4_flash/include/sparkpipe/llm_defines.h (the single
parameter carrier the common modules consume) to that contract so header,
contract and checkpoint config stay in lockstep. The C compiler evaluates
every macro, including the per-layer GDN predicate.
Run: python3 tests/test_qwen4_flash_model_header.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from c_macro_values import MacroProbeError, c_macro_values, value_matches

REPOSITORY = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORIES = [REPOSITORY / "model-families/qwen4_flash/include", REPOSITORY,
                       REPOSITORY / "include"]
CONTRACT = REPOSITORY / "model_contracts/qwen4_flash_authoritative.json"

# contract section/key -> header macro
BINDINGS = {
    ("model", "hidden_dimension"): "SPARK_LLM_HIDDEN_DIMENSION",
    ("model", "layer_count"): "SPARK_LLM_LAYER_COUNT",
    ("model", "vocabulary_size"): "SPARK_LLM_VOCAB_COUNT",
    ("model", "attention_head_count"): "SPARK_LLM_ATTN_HEAD_COUNT",
    ("model", "kv_head_count"): "SPARK_LLM_KV_HEAD_COUNT",
    ("model", "head_dimension"): "SPARK_LLM_HEAD_DIMENSION",
    ("model", "mtp_layer_count"): "SPARK_LLM_MODEL_MTP_LAYER_COUNT",
    ("model", "maximum_context_tokens"): "SPARK_LLM_MAXIMUM_CONTEXT_TOKENS",
    ("hybrid_attention", "period"): "SPARK_LLM_ATTN_PERIOD",
    ("hybrid_attention", "full_phase"): "SPARK_LLM_FULL_ATTENTION_PHASE",
    ("hybrid_attention", "linear_layer_count"): "SPARK_LLM_GDN_LAYER_COUNT",
    ("hybrid_attention", "full_layer_count"): "SPARK_LLM_FULL_ATTENTION_LAYER_COUNT",
    ("linear_attn", "key_head_count"): "SPARK_LLM_GDN_KEY_HEAD_COUNT",
    ("linear_attn", "value_head_count"): "SPARK_LLM_GDN_VALUE_HEAD_COUNT",
    ("linear_attn", "key_dimension"): "SPARK_LLM_GDN_HEAD_KEY_DIMENSION",
    ("linear_attn", "value_dimension"): "SPARK_LLM_GDN_HEAD_VALUE_DIMENSION",
    ("linear_attn", "short_conv_kernel"): "SPARK_LLM_GDN_CONV_KERNEL",
    ("attention", "rope_dimension"): "SPARK_LLM_ROPE_DIMENSION",
    ("attention", "rope_theta"): "SPARK_LLM_ROPE_THETA",
    ("moe", "routed_expert_count"): "SPARK_LLM_ROUTED_EXPERT_COUNT",
    ("moe", "experts_per_token"): "SPARK_LLM_EXPERTS_PER_TOKEN",
    ("moe", "shared_expert_count"): "SPARK_LLM_SHARED_EXPERT_COUNT",
    ("moe", "expert_intermediate_dimension"): "SPARK_LLM_EXPERT_INTERMEDIATE_DIMENSION",
    ("moe", "shared_expert_intermediate_dimension"): "SPARK_LLM_SHARED_EXPERT_INTERMEDIATE_DIMENSION",
    ("hyper_connection", "stream_count"): "SPARK_LLM_HC_STREAM_COUNT",
    ("hyper_connection", "lowrank_dimension"): "SPARK_LLM_HC_LOWRANK_DIMENSION",
    ("indexer", "head_count"): "SPARK_LLM_INDEXER_HEAD_COUNT",
    ("indexer", "kv_head_count"): "SPARK_LLM_INDEXER_KV_HEAD_COUNT",
    ("indexer", "head_dimension"): "SPARK_LLM_INDEXER_HEAD_DIMENSION",
    ("indexer", "budget"): "SPARK_LLM_INDEXER_BUDGET",
    ("indexer", "compress_ratio"): "SPARK_LLM_INDEXER_COMPRESS_RATIO",
    ("ple", "layer_index_weights"): "SPARK_LLM_PLE_LAYER_INDEX",
    ("ple", "ngram_size"): "SPARK_LLM_PLE_NGRAM_SIZE",
    ("ple", "heads_per_ngram"): "SPARK_LLM_PLE_HEADS_PER_NGRAM",
    ("ple", "shard_count"): "SPARK_LLM_PLE_SHARD_COUNT",
    ("ple", "conv_kernel"): "SPARK_LLM_PLE_CONV_KERNEL",
}


def composed_expectations(contract: dict) -> dict[str, int]:
    model = contract["model"]
    linear = contract["linear_attn"]
    qk_dimension = linear["key_head_count"] * linear["key_dimension"]
    value_dimension = linear["value_head_count"] * linear["value_dimension"]
    return {
        "SPARK_LLM_GDN_VALUE_HEADS_PER_KEY_HEAD": linear["value_head_count"] // linear["key_head_count"],
        "SPARK_LLM_GDN_QK_DIMENSION": qk_dimension,
        "SPARK_LLM_GDN_VALUE_DIMENSION": value_dimension,
        "SPARK_LLM_GDN_CONV_CHANNELS": 2 * qk_dimension + value_dimension,
        "SPARK_LLM_ATTN_QUERY_DIMENSION": model["attention_head_count"] * model["head_dimension"],
        "SPARK_LLM_ATTN_KV_DIMENSION": model["kv_head_count"] * model["head_dimension"],
        "SPARK_LLM_ATTN_CACHE_TOKEN_ELEMENTS": 2 * model["kv_head_count"] * model["head_dimension"],
        "SPARK_LLM_HC_STREAM_WIDTH": contract["hyper_connection"]["stream_count"] * model["hidden_dimension"],
    }


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    layer_count = contract["model"]["layer_count"]
    composed = composed_expectations(contract)
    gdn_flags = [f"SPARK_LLM_LAYER_IS_GDN({layer}u)" for layer in range(layer_count)]
    try:
        values = c_macro_values(INCLUDE_DIRECTORIES, "sparkpipe/llm_defines.h",
                                list(BINDINGS.values()) + list(composed) + gdn_flags,
                                ("SPARK_LLM_MTP_LAYER_COUNT=0u",))
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
    period = values["SPARK_LLM_ATTN_PERIOD"]
    phase = values["SPARK_LLM_FULL_ATTENTION_PHASE"]
    gdn_layers = [layer for layer, flag in enumerate(gdn_flags) if values[flag] != 0.0]
    if gdn_layers != [layer for layer in range(layer_count) if layer % period != phase]:
        failures.append(f"LAYER_IS_GDN marks layers {gdn_layers}")
    if len(gdn_layers) != values["SPARK_LLM_GDN_LAYER_COUNT"] or \
            layer_count - len(gdn_layers) != values["SPARK_LLM_FULL_ATTENTION_LAYER_COUNT"]:
        failures.append(f"{len(gdn_layers)} GDN layers disagree with the header layer counts")
    if values["SPARK_LLM_GDN_VALUE_HEAD_COUNT"] % values["SPARK_LLM_GDN_KEY_HEAD_COUNT"]:
        failures.append("linear value heads must group evenly onto key heads")
    if values["SPARK_LLM_EXPERT_INTERMEDIATE_DIMENSION"] % 128 or values["SPARK_LLM_HIDDEN_DIMENSION"] % 128:
        failures.append("expert geometry must tile 128-block FP8 scales")
    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        print(f"FAILED {len(failures)} binding(s)")
        return 1
    print(f"PASS qwen4_flash header matches the authoritative contract ({len(BINDINGS)} bindings + "
          f"{len(composed)} composed + {layer_count} layer kinds, evaluated by the C compiler)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
