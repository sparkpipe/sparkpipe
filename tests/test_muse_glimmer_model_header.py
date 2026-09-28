#!/usr/bin/env python3
"""Bind the muse_glimmer geometry header to the authoritative contract.

The contract (model_contracts/muse_glimmer_authoritative.json) pins the HF
publisher config and the modeling source commit; this test binds
model-families/muse_glimmer/include/sparkpipe/spark_muse_glimmer_model.h to
that contract so header, contract and checkpoint config stay in lockstep.
Every macro is evaluated by the C compiler, so unsigned integer division and
float suffixes behave exactly as they do in the driver; the TP16 per-rank
geometry is checked for every rank and the layer-kind predicate for every
layer.
Run: python3 tests/test_muse_glimmer_model_header.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from c_macro_values import MacroProbeError, c_macro_values, value_matches

REPOSITORY = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORIES = [REPOSITORY / "model-families/muse_glimmer/include"]
CONTRACT = REPOSITORY / "model_contracts/muse_glimmer_authoritative.json"

TP_DEGREE = 16

BINDINGS = {
    ("model", "hidden_dimension"): "SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION",
    ("model", "layer_count"): "SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT",
    ("model", "vocabulary_size"): "SPARK_MUSE_GLIMMER_MODEL_VOCAB_COUNT",
    ("model", "attention_head_count"): "SPARK_MUSE_GLIMMER_MODEL_ATTENTION_HEAD_COUNT",
    ("model", "kv_head_count"): "SPARK_MUSE_GLIMMER_MODEL_KV_HEAD_COUNT",
    ("model", "head_dimension"): "SPARK_MUSE_GLIMMER_MODEL_HEAD_DIMENSION",
    ("model", "intermediate_dimension"): "SPARK_MUSE_GLIMMER_MODEL_INTERMEDIATE_DIMENSION",
    ("model", "mtp_layer_count"): "SPARK_MUSE_GLIMMER_MODEL_MTP_LAYER_COUNT",
    ("model", "maximum_context_tokens"): "SPARK_MUSE_GLIMMER_MODEL_MAXIMUM_CONTEXT_TOKENS",
    ("model", "rms_norm_epsilon"): "SPARK_MUSE_GLIMMER_MODEL_RMS_NORM_EPSILON",
    ("model", "post_norm_epsilon"): "SPARK_MUSE_GLIMMER_MODEL_POST_NORM_EPSILON",
    ("model", "sliding_window"): "SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW",
    ("model", "qk_scale_factor"): "SPARK_MUSE_GLIMMER_MODEL_ATTN_QK_SCALE_FACTOR",
    ("model", "output_multiplier"): "SPARK_MUSE_GLIMMER_MODEL_OUTPUT_MULTIPLIER",
    ("model", "final_logit_softcapping"): "SPARK_MUSE_GLIMMER_MODEL_FINAL_LOGIT_SOFTCAP",
    ("model", "bos_token_id"): "SPARK_MUSE_GLIMMER_MODEL_BOS_TOKEN_ID",
    ("model", "eos_token_id"): "SPARK_MUSE_GLIMMER_MODEL_EOS_TOKEN_ID",
    ("hybrid_attention", "period"): "SPARK_MUSE_GLIMMER_MODEL_ATTENTION_PERIOD",
    ("hybrid_attention", "full_phase"): "SPARK_MUSE_GLIMMER_MODEL_FULL_ATTENTION_PHASE",
    ("hybrid_attention", "sliding_layer_count"): "SPARK_MUSE_GLIMMER_MODEL_SLIDING_LAYER_COUNT",
    ("hybrid_attention", "full_layer_count"): "SPARK_MUSE_GLIMMER_MODEL_FULL_ATTENTION_LAYER_COUNT",
    ("attention", "rope_dimension"): "SPARK_MUSE_GLIMMER_MODEL_ATTN_ROPE_DIMENSION",
    ("attention", "rope_theta"): "SPARK_MUSE_GLIMMER_MODEL_ATTN_ROPE_THETA",
    ("attention", "attention_scale"): "SPARK_MUSE_GLIMMER_MODEL_ATTN_SCALE",
}


def composed_expectations(model: dict) -> dict[str, int]:
    heads = model["attention_head_count"]
    kv_heads = model["kv_head_count"]
    head_dimension = model["head_dimension"]
    local_heads = heads // TP_DEGREE
    local_kv_heads = kv_heads // min(TP_DEGREE, kv_heads)
    prefix = "SPARK_MUSE_GLIMMER_MODEL_"
    expected = {
        prefix + "ATTN_QUERY_DIMENSION": heads * head_dimension,
        prefix + "ATTN_KV_DIMENSION": kv_heads * head_dimension,
        prefix + "ATTN_QUERY_GROUP": heads // kv_heads,
        prefix + "ATTN_CACHE_TOKEN_ELEMENTS": 2 * kv_heads * head_dimension,
        prefix + f"KV_SHARD_COUNT({TP_DEGREE}u)": min(TP_DEGREE, kv_heads),
        prefix + f"ATTN_LOCAL_QUERY_HEAD_COUNT({TP_DEGREE}u)": local_heads,
        prefix + f"ATTN_LOCAL_KV_HEAD_COUNT({TP_DEGREE}u)": local_kv_heads,
        prefix + f"ATTN_LOCAL_QUERY_DIMENSION({TP_DEGREE}u)": local_heads * head_dimension,
        prefix + f"ATTN_LOCAL_QUERY_GATE_DIMENSION({TP_DEGREE}u)": local_heads * 2 * head_dimension,
        prefix + f"ATTN_LOCAL_KV_DIMENSION({TP_DEGREE}u)": local_kv_heads * head_dimension,
        prefix + f"QGKV_LOCAL_ROWS({TP_DEGREE}u)":
            local_heads * 2 * head_dimension + 2 * local_kv_heads * head_dimension,
        prefix + f"MLP_LOCAL_INTERMEDIATE({TP_DEGREE}u)": model["intermediate_dimension"] // TP_DEGREE,
        prefix + f"VOCAB_LOCAL_ROWS({TP_DEGREE}u)": model["vocabulary_size"] // TP_DEGREE,
        prefix + "HIDDEN_BF16_BYTES": model["hidden_dimension"] * 2,
    }
    for rank in range(TP_DEGREE):
        expected[prefix + f"ATTN_RANK_KV_HEAD_BASE({TP_DEGREE}u, {rank}u)"] = rank * kv_heads // TP_DEGREE
    return expected


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    model = contract["model"]
    prefix = "SPARK_MUSE_GLIMMER_MODEL_"
    composed = composed_expectations(model)
    layer_flags = [prefix + f"LAYER_IS_FULL_ATTENTION({layer}u)" for layer in range(model["layer_count"])]
    try:
        values = c_macro_values(INCLUDE_DIRECTORIES, "sparkpipe/spark_muse_glimmer_model.h",
                                list(BINDINGS.values()) + list(composed) + layer_flags)
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
    full_layers = [layer for layer, flag in enumerate(layer_flags) if values[flag] != 0.0]
    period = values[prefix + "ATTENTION_PERIOD"]
    phase = values[prefix + "FULL_ATTENTION_PHASE"]
    if full_layers != [layer for layer in range(model["layer_count"]) if layer % period == phase]:
        failures.append(f"LAYER_IS_FULL_ATTENTION marks layers {full_layers}")
    if len(full_layers) != values[prefix + "FULL_ATTENTION_LAYER_COUNT"]:
        failures.append(f"{len(full_layers)} layers are full attention but the header counts "
                        f"{values[prefix + 'FULL_ATTENTION_LAYER_COUNT']}")
    if (values[prefix + "SLIDING_LAYER_COUNT"] + values[prefix + "FULL_ATTENTION_LAYER_COUNT"]
            != values[prefix + "LAYER_COUNT"]):
        failures.append("sliding and full attention layer counts do not cover the stack")
    for name, whole in ((f"MLP_LOCAL_INTERMEDIATE({TP_DEGREE}u)", "INTERMEDIATE_DIMENSION"),
                        (f"VOCAB_LOCAL_ROWS({TP_DEGREE}u)", "VOCAB_COUNT"),
                        (f"ATTN_LOCAL_QUERY_HEAD_COUNT({TP_DEGREE}u)", "ATTENTION_HEAD_COUNT")):
        if values[prefix + name] * TP_DEGREE != values[prefix + whole]:
            failures.append(f"{name} x {TP_DEGREE} ranks does not cover {whole}")
    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        print(f"FAILED {len(failures)} binding(s)")
        return 1
    print(f"PASS muse_glimmer header matches the authoritative contract ({len(BINDINGS)} bindings + "
          f"{len(composed)} composed + {len(layer_flags)} layer kinds, evaluated by the C compiler)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
