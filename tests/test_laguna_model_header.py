#!/usr/bin/env python3
"""Bind every macro in spark_laguna_model.h and its single-source
llm_defines.h against model_contracts/laguna_authoritative.json, and prove
the hybrid-layer and head-count derivations for all 48 layers.

Pure python; runs anywhere. RED on any mismatch: llm_defines.h is the single
parameter source, the family header is its alias shim, and this test is the
census lock on both.
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "model-families/laguna/include/sparkpipe/spark_laguna_model.h"
LLM_DEFINES = ROOT / "model-families/laguna/include/sparkpipe/llm_defines.h"
CONTRACT = ROOT / "model_contracts/laguna_authoritative.json"
NAME_PATTERN = r"(?:SPARK_LAGUNA_MODEL_|SPARK_LLM_)[A-Z0-9_]+"


def define_values(text, prefix_pattern):
    values = {}
    for match in re.finditer(r"#define (" + prefix_pattern + r")\s+(.+?)(?=\n(?:#|\n))", text, re.S):
        name, body = match.group(1), match.group(2).strip()
        body = body.split("\\")[0].strip()
        values[name] = body
    return values


def resolve(expression, values, depth=0):
    if depth > 32:
        raise RuntimeError("macro cycle")
    def substitute(match):
        name = match.group(0)
        if name not in values:
            raise KeyError(name)
        return "(" + values[name] + ")"
    previous = None
    current = expression
    while previous != current:
        previous = current
        current = re.sub(NAME_PATTERN, substitute, current)
    python_literal = re.sub(r"\b([0-9]+)[uU]f?\b", r"\1", current.replace("f", ""))
    python_literal = re.sub(r"\s+", " ", python_literal).strip()
    while python_literal.startswith("(") and matching_paren(python_literal, 0) == len(python_literal) - 1:
        python_literal = python_literal[1:-1].strip()
    depth = 0
    question = colon = -1
    for offset, char in enumerate(python_literal):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "?" and depth == 0:
            question = offset
        elif char == ":" and depth == 0 and question >= 0 and colon < 0:
            colon = offset
    if question >= 0:
        condition = python_literal[:question].strip()
        consequent = python_literal[question + 1:colon].strip()
        alternative = python_literal[colon + 1:].strip()
        python_literal = f"({consequent} if {condition} else {alternative})"
    return eval(python_literal, {"__builtins__": {}}, {})


def matching_paren(text, start):
    depth = 0
    for offset, char in enumerate(text[start:], start):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return offset
    return -1


def parameterized_macro_body(normalized, name, parameter):
    match = re.search(r"#define " + re.escape(name) + r"\(" + parameter + r"\)\s+(.+?)(?=\n#|\n\n)", normalized)
    if not match:
        raise AssertionError(f"missing macro {name}({parameter})")
    return match.group(1).strip()


def main():
    contract = json.loads(CONTRACT.read_text())
    config = contract["source"]["config_json"]
    text = HEADER.read_text()
    llm_text = LLM_DEFINES.read_text()
    normalized = text.replace("\\\n", " ")
    values = define_values(normalized, r"SPARK_LAGUNA_MODEL_[A-Z0-9_]+")
    values.update(define_values(llm_text.replace("\\\n", " "), r"SPARK_LLM_[A-Z0-9_]+"))
    failures = []

    def check(label, actual, expected):
        if actual != expected:
            failures.append(f"{label}: header {actual!r} != contract {expected!r}")

    check("hidden", resolve("SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION", values), config["hidden_size"])
    check("layers", resolve("SPARK_LAGUNA_MODEL_LAYER_COUNT", values), config["num_hidden_layers"])
    check("vocab", resolve("SPARK_LAGUNA_MODEL_OUTPUT_VOCAB_COUNT", values), config["vocab_size"])
    check("context", resolve("SPARK_LAGUNA_MODEL_MAXIMUM_CONTEXT_TOKENS", values), config["max_position_embeddings"])
    check("kv_heads", resolve("SPARK_LAGUNA_MODEL_ATTENTION_KV_HEAD_COUNT", values), config["num_key_value_heads"])
    check("head_dim", resolve("SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION", values), config["head_dim"])
    check("window", resolve("SPARK_LAGUNA_MODEL_SLIDING_WINDOW", values), config["sliding_window"])
    check("q_full", resolve("SPARK_LAGUNA_MODEL_ATTENTION_Q_HEAD_COUNT_FULL", values), config["num_attention_heads_per_layer_full"])
    check("q_sliding", resolve("SPARK_LAGUNA_MODEL_ATTENTION_Q_HEAD_COUNT_SLIDING", values), config["num_attention_heads_per_layer_sliding"])
    check("experts", resolve("SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT", values), config["moe"]["num_experts"])
    check("top_k", resolve("SPARK_LAGUNA_MODEL_MOE_TOP_K", values), config["moe"]["num_experts_per_tok"])
    check("moe_inter", resolve("SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION", values), config["moe"]["moe_intermediate_size"])
    check("routed_scale", resolve("SPARK_LAGUNA_MODEL_MOE_ROUTED_SCALING_FACTOR", values), config["moe"]["routed_scaling_factor"])
    check("dense_inter", resolve("SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION", values), config["intermediate_size"])
    check("eos_count", resolve("SPARK_LAGUNA_MODEL_END_OF_TEXT_TOKEN_COUNT", values), len(config["eos_token_ids"]))
    check("eos0", resolve("SPARK_LAGUNA_MODEL_END_OF_TEXT_TOKEN_ID", values), config["eos_token_ids"][0])
    check("eos1", resolve("SPARK_LAGUNA_MODEL_END_OF_TURN_TOKEN_ID", values), config["eos_token_ids"][1])
    check("pad", resolve("SPARK_LAGUNA_MODEL_PAD_TOKEN_ID", values), config["pad_token_id"])

    scale = resolve("SPARK_LAGUNA_MODEL_ATTENTION_SCALE", values)
    if abs(scale - 128.0 ** -0.5) > 1e-15:
        failures.append(f"attention scale {scale!r} is not 128^-0.5")

    yarn = contract["yarn"]
    check("rope_full_theta", resolve("SPARK_LAGUNA_MODEL_ROPE_FULL_THETA", values), int(yarn["theta"]))
    check("rope_full_factor", resolve("SPARK_LAGUNA_MODEL_ROPE_FULL_FACTOR", values), int(yarn["factor"]))
    check("rope_full_orig", resolve("SPARK_LAGUNA_MODEL_ROPE_FULL_ORIGINAL_POSITIONS", values), int(yarn["original_positions"]))
    check("rope_full_beta_fast", resolve("SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_FAST", values), int(yarn["beta_fast"]))
    check("rope_full_beta_slow", resolve("SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_SLOW", values), int(yarn["beta_slow"]))
    factor_header = resolve("SPARK_LAGUNA_MODEL_ROPE_FULL_ATTENTION_FACTOR", values)
    if abs(factor_header - yarn["attention_factor"]) > 1e-15:
        failures.append(f"attention factor {factor_header!r} != stated {yarn['attention_factor']!r}")
    if abs(yarn["attention_factor"] - (0.1 * math.log(yarn["factor"]) + 1.0)) > 1e-15:
        failures.append("stated attention factor is not 0.1*ln(factor)+1; re-check the contract")
    rot_full = re.search(r"#define SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION\s*\(\(uint32_t\)\(SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION \*\s*SPARK_LAGUNA_MODEL_ROPE_FULL_PARTIAL_ROTARY_FACTOR\)\)", normalized)
    if not rot_full:
        failures.append("ROPE_FULL_ROTARY_DIMENSION is not stated as head_dim * partial_rotary_factor")
    check("rope_full_rot", resolve("SPARK_LAGUNA_MODEL_ROPE_FULL_PARTIAL_ROTARY_FACTOR", values) and int(config["head_dim"] * config["rope_full"]["partial_rotary_factor"]), yarn["rotary_dimension"])
    check("rope_sliding_theta", resolve("SPARK_LAGUNA_MODEL_ROPE_SLIDING_THETA", values), int(config["rope_sliding"]["rope_theta"]))
    rot_sliding = re.search(r"#define SPARK_LAGUNA_MODEL_ROPE_SLIDING_ROTARY_DIMENSION\s*\(\(uint32_t\)\(SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION \*\s*SPARK_LAGUNA_MODEL_ROPE_SLIDING_PARTIAL_ROTARY_FACTOR\)\)", normalized)
    if not rot_sliding:
        failures.append("ROPE_SLIDING_ROTARY_DIMENSION is not stated as head_dim * partial_rotary_factor")
    check("rope_sliding_rot", int(config["head_dim"] * config["rope_sliding"]["partial_rotary_factor"]), 128)

    sliding_macro = parameterized_macro_body(normalized, "SPARK_LAGUNA_MODEL_LAYER_IS_SLIDING", "layer_index")
    heads_macro = parameterized_macro_body(normalized, "SPARK_LAGUNA_MODEL_LAYER_HEAD_COUNT", "layer_index")
    heads_macro = heads_macro.replace("SPARK_LAGUNA_MODEL_LAYER_IS_SLIDING(layer_index)", sliding_macro)
    heads_derived = [int(resolve(heads_macro.replace("layer_index", f"({index})"), values)) for index in range(48)]
    sliding_derived = [int(bool(resolve(sliding_macro.replace("layer_index", f"({index})"), values))) for index in range(48)]
    for index in range(48):
        expected = config["num_attention_heads_per_layer_sliding"] if index % 4 != 0 else config["num_attention_heads_per_layer_full"]
        check(f"layer {index} heads", heads_derived[index], expected)
        check(f"layer {index} sliding", sliding_derived[index], 1 if index % 4 != 0 else 0)
    check("full layer count", resolve("SPARK_LAGUNA_MODEL_FULL_LAYER_COUNT", values), 12)
    check("sliding layer count", resolve("SPARK_LAGUNA_MODEL_SLIDING_LAYER_COUNT", values), 36)
    if "SPARK_LAGUNA_MODEL_STAGE_LAYER_COUNT(stage_count,stage_index)" not in normalized:
        failures.append("missing SPARK_LAGUNA_MODEL_STAGE_LAYER_COUNT(stage_count,stage_index) macro")
    for stage_count, expected in ((2, contract["fleet_shape"]["stage_layer_counts"][0]), (4, 12)):
        if resolve("SPARK_LAGUNA_MODEL_LAYER_COUNT", values) // stage_count != expected:
            failures.append(f"stage span wrong for {stage_count} stages")

    census = contract["checkpoint_census"]
    tp = json.loads((ROOT / "model-families/laguna/tensor_patterns.json").read_text())
    check("census patterns", tp["tensor_pattern_count"], census["pattern_count"])
    check("census tensors", tp["tensor_count"], census["tensor_count"])

    if failures:
        for failure in failures:
            print("FAIL", failure)
        return 1
    print("laguna model header binds the contract: geometry, both rope regimes, 48-layer hybrid derivation, census, tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
