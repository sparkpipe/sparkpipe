#!/usr/bin/env python3
"""Bind both gemma4 variants to their authoritative contracts.

One family, one parameter file: model-families/gemma4/include/sparkpipe/llm_defines.h
carries the 31B values under SPARK_GEMMA4_MODEL_* and the 26B-A4B values under
SPARK_GEMMA4_MOE_*; the SPARK_GEMMA4_MODEL_* alias fold under
SPARK_GEMMA4_MOE_BUILD re-points the generic namespace at the MoE values at
compile time. Each contract binds to its own prefix in this single file, and
the C compiler evaluates every macro.
Run: python3 tests/test_gemma4_model_header.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from c_macro_values import MacroProbeError, c_macro_values, value_matches

REPOSITORY = Path(__file__).resolve().parents[1]

FAMILY_DEFINES = REPOSITORY / "model-families/gemma4/include/sparkpipe/llm_defines.h"
INCLUDE_DIRECTORIES = [REPOSITORY / "model-families/gemma4/include",
                       REPOSITORY / "model-families/common/include", REPOSITORY,
                       REPOSITORY / "include"]

CONTRACTS = [
    {
        "path": REPOSITORY / "model_contracts/gemma4_31b_authoritative.json",
        "header": FAMILY_DEFINES,
        "prefix": "SPARK_GEMMA4_MODEL_",
    },
    {
        "path": REPOSITORY / "model_contracts/gemma4_26b_a4b_authoritative.json",
        "header": FAMILY_DEFINES,
        "prefix": "SPARK_GEMMA4_MOE_",
    },
]

# contract "model" key -> header macro suffix after the per-contract prefix
BINDINGS = {
    "hidden_dimension": "HIDDEN_DIMENSION",
    "layer_count": "LAYER_COUNT",
    "vocabulary_size": "VOCAB_COUNT",
    "maximum_context_tokens": "MAXIMUM_CONTEXT_TOKENS",
    "sliding_query_head_count": "SLIDING_QUERY_HEAD_COUNT",
    "sliding_kv_head_count": "SLIDING_KV_HEAD_COUNT",
    "sliding_head_dimension": "SLIDING_HEAD_DIMENSION",
    "full_query_head_count": "FULL_QUERY_HEAD_COUNT",
    "full_kv_head_count": "FULL_KV_HEAD_COUNT",
    "full_head_dimension": "FULL_HEAD_DIMENSION",
    "sliding_window_tokens": "SLIDING_WINDOW_TOKENS",
    "full_layer_period": "FULL_LAYER_PERIOD",
    "full_layer_phase": "FULL_LAYER_PHASE",
    "sliding_rope_theta": "SLIDING_ROPE_THETA",
    "full_rope_base": "FULL_ROPE_BASE",
    "qk_scale": "QK_SCALE",
    "rms_norm_epsilon": "RMS_NORM_EPSILON",
    "embed_scale": "EMBED_SCALE",
    "dense_intermediate_dimension": "DENSE_INTERMEDIATE_DIMENSION",
    "final_logit_softcapping": "FINAL_LOGIT_SOFTCAP",
    "bos_token_id": "BOS_TOKEN_ID",
    "pad_token_id": "PAD_TOKEN_ID",
}

MOE_ONLY_BINDINGS = {
    "routed_expert_count": "ROUTED_EXPERT_COUNT",
    "experts_per_token": "EXPERTS_PER_TOKEN",
    "expert_intermediate_dimension": "EXPERT_INTERMEDIATE_DIMENSION",
}

COMPOSED_BINDINGS = {
    "SLIDING_QUERY_DIMENSION": lambda m: m["sliding_query_head_count"] * m["sliding_head_dimension"],
    "SLIDING_KV_DIMENSION": lambda m: m["sliding_kv_head_count"] * m["sliding_head_dimension"],
    "FULL_QUERY_DIMENSION": lambda m: m["full_query_head_count"] * m["full_head_dimension"],
    "FULL_KV_DIMENSION": lambda m: m["full_kv_head_count"] * m["full_head_dimension"],
    "FULL_LAYER_COUNT": lambda m: m["layer_count"] // m["full_layer_period"],
    "SLIDING_LAYER_COUNT": lambda m: m["layer_count"] - m["layer_count"] // m["full_layer_period"],
    "FULL_ROPE_DIMENSION": lambda m: m["full_head_dimension"],
}


EOS_SUFFIXES = ("EOS_TOKEN_ID", "EOS_ALTERNATE_TOKEN_ID", "EOS_ALTERNATE_2_TOKEN_ID")


def contract_expressions(prefix: str, model: dict) -> list[str]:
    names = [prefix + suffix for suffix in BINDINGS.values()]
    names += [prefix + suffix for key, suffix in MOE_ONLY_BINDINGS.items() if key in model]
    names += [prefix + suffix for suffix in COMPOSED_BINDINGS]
    names += [prefix + suffix for suffix in EOS_SUFFIXES]
    return names


def check_contract(entry: dict, values: dict) -> list[str]:
    prefix = entry["prefix"]
    contract = json.loads(entry["path"].read_text(encoding="utf-8"))
    model = contract["model"]
    tag = contract["model_id"]
    failures: list[str] = []
    for key, suffix in BINDINGS.items():
        expected = model[key]
        actual = values[prefix + suffix]
        if not value_matches(expected, actual):
            failures.append(f"{tag} {prefix}{suffix}: header {actual!r} contract {expected!r}")
    for key, suffix in MOE_ONLY_BINDINGS.items():
        if key not in model:
            continue
        expected = model[key]
        actual = values[prefix + suffix]
        if not value_matches(expected, actual):
            failures.append(f"{tag} {prefix}{suffix}: header {actual!r} contract {expected!r}")
    for suffix, derive in COMPOSED_BINDINGS.items():
        expected = derive(model)
        actual = values[prefix + suffix]
        if not value_matches(expected, actual):
            failures.append(f"{tag} {prefix}{suffix}: header {actual!r} derived {expected!r}")
    eos = model["eos_token_ids"]
    if values[prefix + "EOS_TOKEN_ID"] != float(eos[0]):
        failures.append(f"{tag}: {prefix}EOS_TOKEN_ID {values[prefix + 'EOS_TOKEN_ID']!r} "
                        f"is not the first contract eos {eos[0]}")
    header_eos = {values[prefix + suffix] for suffix in EOS_SUFFIXES}
    if header_eos != {float(token) for token in eos}:
        failures.append(f"{tag}: header eos set {sorted(header_eos)} != contract {eos}")
    if contract["source_revision"] == "pending-warm-download":
        failures.append(f"{tag}: source_revision still pending-warm-download")
    elif not re.fullmatch(r"[0-9a-f]{40}", contract["source_revision"]):
        failures.append(f"{tag}: source_revision {contract['source_revision']} is not a 40-hex pin")
    if not model["attention_k_eq_v"]:
        failures.append(f"{tag}: full layers must carry attention_k_eq_v")
    rotated = model["full_rotated_pair_count"]
    expected_rotated = int(model["partial_rotary_factor"] * model["full_head_dimension"] // 2)
    if rotated != expected_rotated or rotated != 64:
        failures.append(f"{tag}: rotated pairs {rotated} != publisher {expected_rotated} (must be 64)")
    return failures


def main() -> int:
    if not all(entry["path"].exists() for entry in CONTRACTS):
        missing = [str(entry["path"]) for entry in CONTRACTS if not entry["path"].exists()]
        print(f"FAILED missing contracts: {missing}")
        return 1
    failures: list[str] = []
    bindings = 0
    expressions: list[str] = []
    for entry in CONTRACTS:
        model = json.loads(entry["path"].read_text(encoding="utf-8"))["model"]
        expressions += contract_expressions(entry["prefix"], model)
    try:
        values = c_macro_values(INCLUDE_DIRECTORIES, "sparkpipe/llm_defines.h", expressions)
    except MacroProbeError as error:
        print(f"FAILED {error}")
        return 1
    for entry in CONTRACTS:
        contract = json.loads(entry["path"].read_text(encoding="utf-8"))
        model = contract["model"]
        bindings += len(BINDINGS) + sum(1 for k in MOE_ONLY_BINDINGS if k in model) + len(COMPOSED_BINDINGS)
        failures.extend(check_contract(entry, values))
        digest = contract.get("digest_freeze", {})
        files = digest.get("files", {})
        for name in ("config.json", "model.safetensors.index.json"):
            if name not in files or not re.fullmatch(r"[0-9a-f]{64}", files[name].get("sha256", "")):
                failures.append(f"{contract['model_id']}: digest_freeze missing {name} sha256")
        shards = [n for n in files if n.endswith(".safetensors")]
        if not shards:
            failures.append(f"{contract['model_id']}: digest_freeze has no shard shas")
    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        print(f"FAILED {len(failures)} binding(s)")
        return 1
    print(f"PASS gemma4 headers match the authoritative contracts ({bindings} bindings over 2 contracts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
