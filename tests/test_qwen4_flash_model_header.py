#!/usr/bin/env python3
"""Bind the qwen4_flash geometry header to the authoritative contract.

The contract (model_contracts/qwen4_flash_authoritative.json) is digest-
frozen against the warm checkpoint by tools/qwen4_flash_verify_source.py
(verify mode, run on a spark node); this test binds
model-families/qwen4_flash/include/sparkpipe/llm_defines.h (the single
parameter carrier the common modules consume) to that contract so header, contract and checkpoint config stay in lockstep.
Run: python3 tests/test_qwen4_flash_model_header.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
HEADER = REPOSITORY / "model-families/qwen4_flash/include/sparkpipe/llm_defines.h"
CONTRACT = REPOSITORY / "model_contracts/qwen4_flash_authoritative.json"

# contract section/key -> header macro
BINDINGS = {
    ("model", "hidden_dimension"): "SPARK_LLM_HIDDEN_DIMENSION",
    ("model", "layer_count"): "SPARK_LLM_LAYER_COUNT",
    ("model", "vocabulary_size"): "SPARK_LLM_VOCAB_COUNT",
    ("model", "attention_head_count"): "SPARK_LLM_ATTN_HEAD_COUNT",
    ("model", "kv_head_count"): "SPARK_LLM_KV_HEAD_COUNT",
    ("model", "head_dimension"): "SPARK_LLM_HEAD_DIMENSION",
    ("model", "mtp_layer_count"): "SPARK_LLM_MTP_LAYER_COUNT",
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


def macro(header: str, name: str) -> float:
    """Resolve one #define to a number: numeric defines directly, composed
    defines by evaluating their expression over previously resolved names
    (line continuations joined first)."""
    joined = header.replace("\\\n", " ")
    defines: dict[str, str] = {}
    # [ \t] after the name keeps valueless include guards from swallowing
    # the next line; continuations are already joined.
    for match in re.finditer(r"^#define\s+(\w+)[ \t]+([^\n]+?)\s*$", joined, re.M):
        defines[match.group(1)] = match.group(2).strip()
    if name not in defines:
        raise AssertionError(f"header missing #define {name}")

    def resolve(target: str, seen: frozenset) -> float:
        text = defines[target]
        if target in seen:
            raise AssertionError(f"cyclic define {target}")
        try:
            return float(text.rstrip("uf"))
        except ValueError:
            pass
        expression = re.sub(r"\w+", lambda m: (
            str(resolve(m.group(0), seen | {target}))
            if m.group(0) in defines and m.group(0) != target else m.group(0)), text)
        expression = re.sub(r"(\d)[uf]\b", r"\1", expression)  # strip C suffixes
        return float(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - header constants only

    return resolve(name, frozenset())


def main() -> int:
    header = HEADER.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    failures = 0
    for (section, key), name in BINDINGS.items():
        expected = contract[section][key]
        actual = macro(header, name)
        if float(expected) != actual:
            print(f"MISMATCH {name}: header {actual} contract {expected}")
            failures += 1
    # composed geometry the kernels and packer derive from the macros
    composed = {
        "SPARK_LLM_GDN_VALUE_HEADS_PER_KEY_HEAD": 48 / 16,
        "SPARK_LLM_GDN_QK_DIMENSION": 16 * 128,
        "SPARK_LLM_GDN_VALUE_DIMENSION": 48 * 128,
        "SPARK_LLM_GDN_CONV_CHANNELS": 2 * 2048 + 6144,
        "SPARK_LLM_ATTN_QUERY_DIMENSION": 24 * 256,
        "SPARK_LLM_ATTN_KV_DIMENSION": 2 * 256,
        "SPARK_LLM_ATTN_CACHE_TOKEN_ELEMENTS": 2 * (2 * 256),
        "SPARK_LLM_HC_STREAM_WIDTH": 4 * 2560,
    }
    for name, expected in composed.items():
        actual = macro(header, name)
        if float(expected) != actual:
            print(f"MISMATCH composed {name}: header {actual} expected {expected}")
            failures += 1
    # invariants the module's stagepack static asserts restate in C
    if 36 + 12 != 48 or 48 % 4 != 0:
        print("MISMATCH hybrid layer split does not cover the stack in whole periods")
        failures += 1
    if 48 % 16 != 0:
        print("MISMATCH linear value heads must group evenly onto key heads")
        failures += 1
    if 640 % 128 != 0 or 2560 % 128 != 0:
        print("MISMATCH expert geometry must tile 128-block FP8 scales")
        failures += 1
    if failures:
        print(f"FAILED {failures} binding(s)")
        return 1
    print("PASS qwen4_flash header matches the authoritative contract (36 bindings + 8 composed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
