#!/usr/bin/env python3
"""Bind the muse_glimmer geometry header to the authoritative contract.

The contract (model_contracts/muse_glimmer_authoritative.json) pins the HF
publisher config and the modeling source commit; this test binds
model-families/muse_glimmer/include/sparkpipe/spark_muse_glimmer_model.h to
that contract so header, contract and checkpoint config stay in lockstep.
Run: python3 tests/test_muse_glimmer_model_header.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
HEADER = REPOSITORY / "model-families/muse_glimmer/include/sparkpipe/spark_muse_glimmer_model.h"
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


def unwrap(expression: str) -> str:
    """Drop enclosing parentheses that wrap the entire expression so the
    ternary scan sees balanced branches."""
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        for index, character in enumerate(expression):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0 and index != len(expression) - 1:
                    return expression
        expression = expression[1:-1].strip()
    return expression


def ternary_to_python(expression: str) -> str:
    """C constant ternaries (the donor KV_SHARD_COUNT form) become python
    conditionals; one '?' per expression, split at the depth-zero ':'."""
    if "?" not in expression:
        return expression
    assert expression.count("?") == 1, f"unsupported ternary chain: {expression}"
    expression = unwrap(expression)
    question = expression.index("?")
    depth = 0
    for index in range(question + 1, len(expression)):
        character = expression[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == ":" and depth == 0:
            condition = expression[:question].strip()
            return (f"(({expression[question + 1:index]}) "
                f"if ({condition}) else ({expression[index + 1:]}))")
    raise AssertionError(f"ternary without colon: {expression}")


def macro(header: str, name: str, tp_degree: int | None = None) -> float:
    """Resolve one #define to a number: numeric defines directly, composed
    defines by evaluating their expression over previously resolved names
    (line continuations joined first; parameterized macros get the TP degree
    and the boundary rank substituted textually, and invocations of other
    parameterized macros expand before the plain name pass)."""
    joined = header.replace("\\\n", " ")
    defines: dict[str, str] = {}
    parameters: dict[str, list[str] | None] = {}
    for match in re.finditer(
            r"^#define\s+(\w+)(?:\(([^)]*)\))?[ \t]+([^\n]+?)\s*$", joined, re.M):
        defines[match.group(1)] = match.group(3).strip()
        parameters[match.group(1)] = (
            [piece.strip() for piece in match.group(2).split(",")]
            if match.group(2) else None)
    if name not in defines:
        raise AssertionError(f"header missing #define {name}")

    def resolve(target: str, seen: frozenset, arguments: dict[str, str] | None = None) -> float:
        text = defines[target]
        if target in seen:
            raise AssertionError(f"cyclic define {target}")
        if arguments:
            for parameter, argument in arguments.items():
                text = re.sub(r"\b%s\b" % re.escape(parameter), argument, text)
        try:
            return float(text.rstrip("uf"))
        except ValueError:
            pass
        if tp_degree is not None:
            text = re.sub(r"\btp_degree\b", str(tp_degree), text)
            text = re.sub(r"\btp_rank\b", str(tp_degree - 1), text)
        for callee, callee_parameters in parameters.items():
            if not callee_parameters:
                continue
            text = re.sub(
                r"\b%s\(([^()]*)\)" % re.escape(callee),
                lambda call, callee=callee, callee_parameters=callee_parameters, seen=seen: str(resolve(
                    callee, seen | {target},
                    dict(zip(callee_parameters, [piece.strip() for piece in call.group(1).split(",")])))),
                text)
        expression = re.sub(r"\w+", lambda m: (
            str(resolve(m.group(0), seen | {target}))
            if m.group(0) in defines and m.group(0) != target else m.group(0)), text)
        expression = re.sub(r"(\d)[uf]\b", r"\1", expression)  # strip C suffixes
        expression = re.sub(r"\((?:float|double|uint32_t|uint64_t|int32_t|int64_t|unsigned|size_t)\)", "", expression)
        expression = ternary_to_python(expression)
        environment = {"__builtins__": {}, "sqrtf": math.sqrt, "sqrt": math.sqrt}
        return float(eval(expression, environment, {}))  # noqa: S307 - header constants only

    return resolve(name, frozenset())


def main() -> int:
    header = HEADER.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    failures = 0
    for (section, key), name in BINDINGS.items():
        expected = contract[section][key]
        actual = macro(header, name)
        if not math.isclose(float(expected), actual, rel_tol=1e-9):
            print(f"MISMATCH {name}: header {actual} contract {expected}")
            failures += 1
    composed = {
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_DIMENSION": 32 * 128,
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_KV_DIMENSION": 2 * 128,
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_GROUP": 32 / 2,
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_CACHE_TOKEN_ELEMENTS": 2 * (2 * 128),
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_HEAD_COUNT": 32 / 16,
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_HEAD_COUNT": 2 / 2,
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_RANK_KV_HEAD_BASE": 15 * 2 / 16,
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_DIMENSION": 2 * 128,
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_GATE_DIMENSION": 2 * 2 * 128,
        "SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_DIMENSION": 1 * 128,
        "SPARK_MUSE_GLIMMER_MODEL_QGKV_LOCAL_ROWS": 2 * 2 * 128 + 2 * 128,
        "SPARK_MUSE_GLIMMER_MODEL_MLP_LOCAL_INTERMEDIATE": 19968 / 16,
        "SPARK_MUSE_GLIMMER_MODEL_VOCAB_LOCAL_ROWS": 202048 / 16,
        "SPARK_MUSE_GLIMMER_MODEL_HIDDEN_BF16_BYTES": 6656 * 2,
    }
    for name, expected in composed.items():
        actual = macro(header, name, tp_degree=TP_DEGREE)
        if not math.isclose(float(expected), actual, rel_tol=1e-9):
            print(f"MISMATCH composed {name}: header {actual} expected {expected}")
            failures += 1
    if 39 + 13 != 52 or 52 % 4 != 0:
        print("MISMATCH hybrid layer split does not cover the stack in whole periods")
        failures += 1
    if 32 % 2 != 0 or 16 % (32 // 16) != 0:
        print("MISMATCH query heads must group evenly onto kv heads")
        failures += 1
    if 128 % 2 != 0:
        print("MISMATCH rope dimension must pair")
        failures += 1
    if 202048 % 16 != 0 or 19968 % 16 != 0:
        print("MISMATCH vocab and intermediate must shard evenly across 16 ranks")
        failures += 1
    if failures:
        print(f"FAILED {failures} binding(s)")
        return 1
    print(f"PASS muse_glimmer header matches the authoritative contract ({len(BINDINGS)} bindings + {len(composed)} composed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
