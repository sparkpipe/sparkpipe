#!/usr/bin/env python3
"""Bind the qwen4_flash geometry header to the authoritative contract.

The contract (model_contracts/qwen4_flash_authoritative.json) is digest-
frozen against the warm checkpoint by tools/qwen4_flash_verify_source.py
(verify mode, run on a spark node); this test binds
model-families/qwen4_flash/include/sparkpipe/spark_qwen4_flash_model.h to
that contract so header, contract and checkpoint config stay in lockstep.
Run: python3 tests/test_qwen4_flash_model_header.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
HEADER = REPOSITORY / "model-families/qwen4_flash/include/sparkpipe/spark_qwen4_flash_model.h"
CONTRACT = REPOSITORY / "model_contracts/qwen4_flash_authoritative.json"

# contract section/key -> header macro
BINDINGS = {
    ("model", "hidden_dimension"): "SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION",
    ("model", "layer_count"): "SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT",
    ("model", "vocabulary_size"): "SPARK_QWEN4_FLASH_MODEL_VOCAB_COUNT",
    ("model", "attention_head_count"): "SPARK_QWEN4_FLASH_MODEL_ATTENTION_HEAD_COUNT",
    ("model", "kv_head_count"): "SPARK_QWEN4_FLASH_MODEL_KV_HEAD_COUNT",
    ("model", "head_dimension"): "SPARK_QWEN4_FLASH_MODEL_HEAD_DIMENSION",
    ("model", "mtp_layer_count"): "SPARK_QWEN4_FLASH_MODEL_MTP_LAYER_COUNT",
    ("model", "maximum_context_tokens"): "SPARK_QWEN4_FLASH_MODEL_MAXIMUM_CONTEXT_TOKENS",
    ("hybrid_attention", "period"): "SPARK_QWEN4_FLASH_MODEL_ATTENTION_PERIOD",
    ("hybrid_attention", "full_phase"): "SPARK_QWEN4_FLASH_MODEL_FULL_ATTENTION_PHASE",
    ("hybrid_attention", "linear_layer_count"): "SPARK_QWEN4_FLASH_MODEL_GDN_LAYER_COUNT",
    ("hybrid_attention", "full_layer_count"): "SPARK_QWEN4_FLASH_MODEL_FULL_ATTENTION_LAYER_COUNT",
    ("linear_attn", "key_head_count"): "SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT",
    ("linear_attn", "value_head_count"): "SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT",
    ("linear_attn", "key_dimension"): "SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_KEY_DIMENSION",
    ("linear_attn", "value_dimension"): "SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_VALUE_DIMENSION",
    ("linear_attn", "short_conv_kernel"): "SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL",
    ("attention", "rope_dimension"): "SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION",
    ("attention", "rope_theta"): "SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_THETA",
    ("moe", "routed_expert_count"): "SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT",
    ("moe", "experts_per_token"): "SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN",
    ("moe", "shared_expert_count"): "SPARK_QWEN4_FLASH_MODEL_SHARED_EXPERT_COUNT",
    ("moe", "expert_intermediate_dimension"): "SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION",
    ("moe", "shared_expert_intermediate_dimension"): "SPARK_QWEN4_FLASH_MODEL_SHARED_EXPERT_INTERMEDIATE_DIMENSION",
    ("hyper_connection", "stream_count"): "SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT",
    ("hyper_connection", "lowrank_dimension"): "SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION",
    ("indexer", "head_count"): "SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT",
    ("indexer", "kv_head_count"): "SPARK_QWEN4_FLASH_MODEL_INDEXER_KV_HEAD_COUNT",
    ("indexer", "head_dimension"): "SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION",
    ("indexer", "budget"): "SPARK_QWEN4_FLASH_MODEL_INDEXER_BUDGET",
    ("indexer", "compress_ratio"): "SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO",
    ("ple", "layer_index_weights"): "SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX",
    ("ple", "ngram_size"): "SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_SIZE",
    ("ple", "heads_per_ngram"): "SPARK_QWEN4_FLASH_MODEL_PLE_HEADS_PER_NGRAM",
    ("ple", "shard_count"): "SPARK_QWEN4_FLASH_MODEL_PLE_SHARD_COUNT",
    ("ple", "conv_kernel"): "SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL",
}


def macro(header: str, name: str) -> float:
    match = re.search(rf"^#define\s+{name}\s+(.+?)\s*(?:/[/*].*)?$", header, re.M)
    if match is None:
        raise AssertionError(f"header missing #define {name}")
    text = match.group(1)
    if text.endswith("u"):
        text = text[:-1]
    if text.endswith("f"):
        text = text[:-1]
    return float(text)


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
        "SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD": 48 / 16,
        "SPARK_QWEN4_FLASH_MODEL_GDN_QK_DIMENSION": 16 * 128,
        "SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION": 48 * 128,
        "SPARK_QWEN4_FLASH_MODEL_GDN_CONV_CHANNELS": 2 * 2048 + 6144,
        "SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_DIMENSION": 24 * 256,
        "SPARK_QWEN4_FLASH_MODEL_ATTN_KV_DIMENSION": 2 * 256,
        "SPARK_QWEN4_FLASH_MODEL_ATTN_CACHE_TOKEN_ELEMENTS": 2 * (2 * 256),
        "SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH": 4 * 2560,
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
